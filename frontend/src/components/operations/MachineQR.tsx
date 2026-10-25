'use client';
import {useEffect,useState} from 'react';
import Image from 'next/image';
import {operationsDownload} from '@/lib/operations';
export default function MachineQR({assetId,name}:{assetId:string;name:string}){
 const [visible,setVisible]=useState(false),[url,setUrl]=useState(''),[error,setError]=useState('');
 useEffect(()=>{if(!visible)return;let active=true,objectURL='';operationsDownload(`/assets/${encodeURIComponent(assetId)}/qr?origin=${encodeURIComponent(location.origin)}`).then(blob=>{if(active){objectURL=URL.createObjectURL(blob);setUrl(objectURL);setError('');}}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;if(objectURL)URL.revokeObjectURL(objectURL);};},[visible,assetId]);
 return <details className="surface machine-qr" onToggle={e=>{setVisible(e.currentTarget.open);if(!e.currentTarget.open){setUrl('');setError('');}}}><summary>Machine QR label</summary>{visible&&<div className="qr-content">{error?<p role="alert">{error}</p>:url?<><div className="qr-label"><strong>{name}</strong><Image unoptimized src={url} width={240} height={240} alt={`QR access label for ${name}`}/><span>{assetId}</span></div><div><h3>Quick access for field engineers</h3><p>Scan with a phone camera or use Scan machine in the workspace. Sign-in and machine access are required.</p><a className="secondary-button" download={`machine-${assetId}.png`} href={url}>Download QR label</a><p className="muted">The downloaded PNG contains the QR code. Pair it with the machine ID when printing. Localhost links require this computer; phone access needs a reachable deployed site.</p></div></>:<p>Preparing label…</p>}</div>}</details>;
}
