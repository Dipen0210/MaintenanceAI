// Treat scanned content as an identifier, never a URL to navigate blindly.
export function scannedAsset(value:string,origin:string):string{
 const text=value.trim();if(!text)throw new Error('Enter a machine ID or scan a PlantPilot label.');
 if(text.includes('://')||text.startsWith('/')){
  const url=new URL(text,origin);
  if(url.origin!==origin||url.pathname!=='/operations/scan'||url.hash||[...url.searchParams.keys()].some(k=>k!=='asset'))throw new Error('This is not a machine label for this application.');
  const values=url.searchParams.getAll('asset');if(values.length!==1||!values[0]||values[0].length>100)throw new Error('Machine label is invalid.');return values[0];
 }
 if(text.length>100||/[\s/?#]/.test(text))throw new Error('Enter a valid machine ID or a label from this application.');
 return text;
}
