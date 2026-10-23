"""Read-only ingestion adapters. Live connectivity requires separately approved gateways."""
from datetime import timedelta
import csv,io,json,random
from typing import Protocol,Iterable
from backend.core.database import iso,utcnow


class ReadOnlyAdapter(Protocol):
    def records(self) -> Iterable[dict]: ...


class SimulatedAdapter:
    def __init__(self,assets,source='demo-simulator',seed=42,now=None,scenario='mixed'):
        self.assets=assets;self.source=source;self.seed=seed;self.now=now or utcnow();self.scenario=scenario

    def records(self):
        rng=random.Random(self.seed)
        for asset in self.assets:
            # Explicit synthetic conveyor incident, not a physical diagnosis.
            incident=asset['id']=='DEMO-CONVEYOR-001' and self.scenario!='normal'
            critical=(self.scenario=='mixed' and asset['id']=='PLA-PUMP-002') or (self.scenario=='critical' and incident)
            for tag,unit,value in [('temperature','degC',105 if critical else 78 if incident else 45),('vibration_rms','mm/s',14 if critical else 7.2 if incident else 1.5),('current','A',42 if critical else 28 if incident else 12)]:
                yield dict(asset_id=asset['id'],tag_id=tag,timestamp_utc=iso(self.now),
                           value=round(value+rng.uniform(-.2,.2),3),unit=unit,quality='good',
                           source=self.source,operating_state='running')


class CSVAdapter:
    def __init__(self,content:bytes):self.content=content
    def records(self):
        try:
            rows=list(csv.DictReader(io.StringIO(self.content.decode('utf-8-sig'))))
        except (UnicodeDecodeError,csv.Error) as exc:raise ValueError('Invalid UTF-8 CSV') from exc
        if len(rows)>10000:raise ValueError('At most 10000 rows per import')
        return rows


class ParquetAdapter:
    def __init__(self,content:bytes):self.content=content
    def records(self):
        import pandas as pd
        try:
            import pyarrow.parquet as pq
            metadata=pq.ParquetFile(io.BytesIO(self.content)).metadata
            if metadata.num_rows>10000 or metadata.num_columns>16 or sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))>64*1024*1024:
                raise ValueError('Parquet exceeds decoded size limits')
            frame=pd.read_parquet(io.BytesIO(self.content))
        except ImportError as exc:raise RuntimeError('Install the parquet optional dependency') from exc
        except Exception as exc:raise ValueError('Invalid Parquet file') from exc
        if len(frame)>10000:raise ValueError('At most 10000 rows per import')
        return frame.to_dict(orient='records')


class OPCUAReadOnlyAdapter:
    """Receives a gateway-owned connection. Only approved node reads are exposed.

    The gateway owns credentials, certificate validation and OT owner approval.
    This adapter exposes no browsing, writes, method calls or control operation.
    """
    def __init__(self,client,mappings,source,approved=False):
        if not approved:raise ValueError('OT owner approval required')
        self.client=client;self.mappings=mappings;self.source=source

    async def records(self):
        rows=[]
        for mapping in self.mappings:
            node=self.client.get_node(mapping['node_id'])
            sample=await node.read_data_value()
            timestamp=sample.SourceTimestamp
            if timestamp is None:raise ValueError('OPC UA source timestamp is missing')
            rows.append(dict(asset_id=mapping['asset_id'],tag_id=mapping['tag_id'],timestamp_utc=timestamp,
                value=sample.Value.Value,unit=mapping['unit'],quality='good' if sample.StatusCode.is_good() else 'bad',
                source=self.source,operating_state=mapping.get('operating_state','unknown')))
        return rows


class HistorianReadOnlyAdapter:
    """Read through an approved integration gateway, not arbitrary supplied URLs."""
    def __init__(self,client,path,mappings,source,approved=False):
        if not approved:raise ValueError('OT owner approval required')
        if not path.startswith('/telemetry/') or '?' in path or '..' in path:raise ValueError('Unapproved historian route')
        self.client=client;self.path=path;self.mappings=mappings;self.source=source
    def records(self):
        response=self.client.get(self.path);response.raise_for_status()
        rows=[]
        for raw in response.json()['observations']:
            mapping=self.mappings.get(raw['tag_id'])
            if mapping is None:raise ValueError('Unapproved historian tag')
            rows.append(dict(asset_id=mapping['asset_id'],tag_id=mapping['tag_id'],timestamp_utc=raw['timestamp_utc'],
                value=raw['value'],unit=raw['unit'],quality=raw['quality'],source=self.source,
                operating_state=raw['operating_state']))
        return rows
