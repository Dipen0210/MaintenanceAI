# Connect a machine and its sensors

This is a working local integration path, not evidence of a commissioned physical plant. A machine can be registered and receive authenticated measurements without being assigned a predictive model. An actual plant gateway, sensor installation, calibrated limits, approved connectivity and site validation remain necessary.

## Try the simulation

Open `/operations` and use **Run simulation** in the top bar. It starts or pauses only the demo sensor generator. The setting survives restart. Running generates readings immediately and then every ten seconds; pausing preserves history, and sensor readings become stale according to their configured freshness limits. The control is restricted to the demo environment and users with access to every affected simulated asset and source. It never starts, stops or changes a physical machine.

The dashboard and asset directory show **Within limits**, **Warning readings**, **Critical readings**, or **Needs verification / Awaiting readings**. These are current, authorized sensor-limit observations, separate from maintenance assessments and model predictions. “Within limits” is not a certification that equipment is safe or physically healthy. Missing, stale, bad, rejected, or inapplicable evidence cannot produce an all-clear condition.

## Register factory equipment

1. Sign in as a plant administrator. In the local demonstration this is `admin` / `admin-demo-password`.
2. Open **Assets → Add machine**. Enter the equipment-register ID, machine name, type and plant area.
3. Add each sensor. Select a measurement type or enter a custom tag, unit, physical bounds, warning/critical thresholds and freshness interval. Enter site-reviewed values; the wizard intentionally does not invent universal safe thresholds.
4. Review the mappings and commissioning notes. Registration creates an asset, its sensor contracts, a gateway source and a source-specific credential in one transaction. It creates no readings or trained-model binding.
5. Download the gateway configuration while the key is available. The stored key is hashed and cannot be retrieved later. The download contains a secret plus OPC UA configuration placeholders; fill those in on the approved gateway. The key expires after 90 days. **Data sources → Gateway setup & access** supports replacement and revocation.
6. Send actual readings and watch **Connect & verify**. Each mapped sensor must supply a current source timestamp, correct unit, finite in-range value and quality. Open machine monitoring to see the values, trends and condition.
7. Initially only the registering administrator has access to the new asset and source. Grant the intended team both asset and source access through **Administration → People & access**. Appropriate document access is separate.

RFID generally serves identification/localization; it does not substitute for vibration, temperature, motor-current or process measurements. The applicable sensor types and mounting locations depend on the machine. See [Siemens industrial identification](https://www.siemens.com/en-us/products/simatic-ident/) and [SKF condition-monitoring parameters](https://evolution.skf.com/us/staying-out-in-front/).

## Visual machine and sensor map

Equipment Overview and Sensor readings include a shaded, outlined technical illustration. Pump, fan and conveyor illustrations are generic references, not digital twins or CAD models of the installed equipment. Selecting a marker displays its recorded value, unit, quality and original timestamp.

An administrator can upload a real JPEG/PNG photograph (5 MiB, at most 12 megapixels), then select a sensor and click its image location or enter horizontal/vertical percentages. Location labels and coordinates persist. Photos are decoded, resized and re-encoded without original metadata, stored privately and served only to authorized users. Replacing a photo resets marker placement to illustrative defaults so old locations are not silently applied to a different image. User-positioned markers are not a certification of the installation.

## OPC UA reference gateway

OPC UA is the reference industrial interoperability path, not a claim that every factory uses it. The [OPC Foundation](https://opcfoundation.org/about/opc-technologies/opc-ua/) describes its machine-to-machine and enterprise interoperability role.

Use the existing read-only OPC UA adapter through `scripts/opcua_gateway.py` on a gateway host that is authorized to reach both the OPC UA server and the MaintenanceAI API. The bridge reads only the configured node IDs; it performs no browsing, writes or method calls. Native MQTT subscription is not included, although another approved gateway can normalize MQTT readings into the same inbound JSON API.

The downloaded configuration needs these deployment-specific values:

```json
{
  "endpoint": "https://maintenance-api.example/api/gateways/REGISTERED_SOURCE_ID/telemetry",
  "source_id": "REGISTERED_SOURCE_ID",
  "token": "YOUR_ONE_TIME_GATEWAY_KEY",
  "opcua_endpoint": "opc.tcp://APPROVED_PLANT_GATEWAY:4840",
  "ot_owner_approved": true,
  "client_certificate": "/private/gateway/client.der",
  "client_private_key": "/private/gateway/client-key.pem",
  "server_certificate": "/private/gateway/approved-server.der",
  "interval_seconds": 10,
  "mappings": [
    {
      "node_id": "APPROVED_NODE_ID",
      "asset_id": "REGISTERED_MACHINE_ID",
      "tag_id": "temperature",
      "unit": "degC",
      "operating_state": "unknown"
    }
  ]
}
```

These are configuration placeholders, not a real connected plant. Do not mark OT approval until obtained. Remote OPC UA connections require SignAndEncrypt, a client certificate/private key and a pinned server certificate. Configure gateway trust/certificate validity with the plant owner; no remote certificate deployment has been validated here. Optional OPC UA username/password values are referenced through `username_env` and `password_env`, not embedded in endpoint URLs. Remote API endpoints require HTTPS. An explicit `allow_insecure_loopback: true` is supported only for local protocol tests.

The reference bridge defaults operating state to `unknown`. Do not hardcode `running` without an approved, accurate state source; unknown state may correctly prevent an applicable condition assessment. Supply actual state through your gateway integration when available. No unit conversion or waveform-to-RMS conversion is inferred; the measurement must already match the registered contract.

```sh
chmod 600 /private/gateway/maintenanceai-gateway.json
venv/bin/python -m scripts.opcua_gateway --config /private/gateway/maintenanceai-gateway.json --once
venv/bin/python -m scripts.opcua_gateway --config /private/gateway/maintenanceai-gateway.json
```

The bridge preserves OPC UA source timestamps and status codes, skips unchanged source observations within a polling session, and reports accepted/flagged/rejected counts. It exits on a connection error without inventing measurements. Deploy under the site's supervised service with appropriate restart/backoff and monitoring. Remote API addresses, OT credentials and network access are not configured by the local demo launcher. `localhost` refers to the gateway host itself, so use the deployed HTTPS API address for another machine.

## Direct gateway API

POST up to 1,000 normalized scalar records to `/api/gateways/{source_id}/telemetry` with `Authorization: Bearer <gateway-key>`:

```json
{
  "records": [{
    "asset_id": "REGISTERED_MACHINE_ID",
    "tag_id": "temperature",
    "timestamp_utc": "ACTUAL_TIMEZONE_AWARE_SENSOR_TIMESTAMP",
    "value": "REPLACE_WITH_ACTUAL_NUMERIC_VALUE",
    "unit": "degC",
    "quality": "good",
    "source": "REGISTERED_SOURCE_ID",
    "operating_state": "unknown"
  }]
}
```

The example is intentionally non-executable until replaced with actual data. The credential cannot read application data, control equipment, access other sources or bypass configured sensor mappings. Current owner account/source/asset grants, credential expiry and revocation are checked on ingestion. Wrong-source batches are rejected before any writes; invalid individual readings are reported without becoming valid telemetry. Manual file import and simulation cannot overwrite a machine's gateway-mapped sensors. Read-only refers to the physical equipment boundary; validated measurements are persisted internally.

## Verification and limits

Tests exercise the actual loopback OPC UA protocol → authenticated ingestion → persisted machine readings path, not a real OT connection. Browser checks register a machine, send explicitly test-generated records, verify the UI and configure its sensor map. Actual factory integration, predictive model qualification for the specific machine, expert review and deployment validation remain separate requirements. The local SQLite service is not a claim of enterprise-scale capacity, high availability or industrial certification.
