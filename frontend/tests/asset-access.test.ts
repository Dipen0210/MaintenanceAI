import {describe,expect,it} from 'vitest';
import {scannedAsset} from '@/lib/asset-access';
const origin='http://localhost:3000';
describe('machine label parsing',()=>{
 it('accepts a machine ID and same-site label without navigating arbitrary URLs',()=>{
  expect(scannedAsset(' DEMO-CONVEYOR-001 ',origin)).toBe('DEMO-CONVEYOR-001');
  expect(scannedAsset(origin+'/operations/scan?asset=PUMP%201',origin)).toBe('PUMP 1');
  expect(scannedAsset('/operations/scan?asset=DEMO-CONVEYOR-001',origin)).toBe('DEMO-CONVEYOR-001');
 });
 it.each(['','https://other.example/operations/scan?asset=A','//other.example/operations/scan?asset=A',origin+'/operations/admin?asset=A',origin+'/operations/scan?asset=A&asset=B',origin+'/operations/scan?asset=A&token=secret',origin+'/operations/scan?asset=A#x','invalid machine','A/B'] )('rejects malformed or foreign labels: %s',input=>{
  expect(()=>scannedAsset(input,origin)).toThrow();
 });
});
