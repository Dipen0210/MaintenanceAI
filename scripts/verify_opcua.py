"""Real OPC UA protocol test against an ephemeral local server, never plant OT."""
import asyncio,json,socket,os,tempfile
from pathlib import Path
from asyncua import Server,Client,ua
from backend.ingestion.adapters import OPCUAReadOnlyAdapter


async def run():
    # Ask the OS for an unused loopback port; the server binds it immediately.
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    endpoint=f'opc.tcp://127.0.0.1:{port}/test/'
    server=Server();await server.init();server.set_endpoint(endpoint)
    namespace=await server.register_namespace('maintenanceai:local-protocol-test')
    node=await server.nodes.objects.add_variable(namespace,'temperature',42.5)
    # Default node is read-only for clients; no set_writable call.
    async with server:
        async with Client(endpoint) as client:
            adapter=OPCUAReadOnlyAdapter(client,[dict(node_id=node.nodeid.to_string(),asset_id='LOCAL-TEST',tag_id='temperature',unit='degC')],'local-opcua-test',approved=True)
            records=await adapter.records()
            assert records[0]['value']==42.5 and records[0]['quality']=='good'
            assert records[0]['timestamp_utc'] is not None
            assert await node.read_value()==42.5
    # Feed the actual read through permission checks, validation and persistence.
    from backend.core.database import migrate,transaction
    from backend.core.demo import seed_demo
    from backend.ingestion.service import ingest
    old_path=os.environ.get('MAINTENANCEAI_DB')
    with tempfile.TemporaryDirectory() as directory:
        os.environ['MAINTENANCEAI_DB']=str(Path(directory)/'opcua.db')
        try:
            migrate();seed_demo()
            with transaction() as conn:
                conn.execute("INSERT INTO sources VALUES('local-opcua-test','demo-plant','Local protocol test','opcua',1,'configured',NULL)")
                conn.execute("INSERT INTO source_grants VALUES('admin','local-opcua-test')")
                conn.execute("DELETE FROM telemetry WHERE asset_id='DEMO-CONVEYOR-001'")
                conn.execute("UPDATE sensors SET source_id='local-opcua-test' WHERE asset_id='DEMO-CONVEYOR-001' AND tag_id='temperature'")
            records[0]['asset_id']='DEMO-CONVEYOR-001'
            outcomes=ingest(records,dict(id='admin',plant_id='demo-plant',role='administrator'),'local-opcua-test')
            assert outcomes[0]['status']=='accepted'
            with transaction() as conn:
                assert conn.execute("SELECT value FROM telemetry WHERE source='local-opcua-test'").fetchone()[0]==42.5
        finally:
            if old_path is None:os.environ.pop('MAINTENANCEAI_DB',None)
            else:os.environ['MAINTENANCEAI_DB']=old_path
    report={'scope':'Local loopback OPC UA test server only; no real plant connectivity',
            'read_value':42.5,'source_timestamp_present':True,'write_operations':0,'persisted_through_validated_ingestion':True,'status':'passed'}
    Path('docs/evaluation/phase1-opcua.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':asyncio.run(run())
