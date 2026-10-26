export type Supplier={id:string;status:string;version:number;content:{name:string;contact:string;reference:string;notes:string}};
export type Quote={id:string;supplier_id:string;part_id:string;status:string;version:number;limitations:string[];content:{reference:string;currency:string;unit:string;unit_price_minor:number;shipping_minor:number;tax_minor:number;minimum_quantity:number;maximum_quantity:number;lead_time_days:number;valid_until:string}};
export type QuoteOption={quote_id:string;supplier_name:string;currency:string;quantity:number;unit:string;total_minor:number;lead_time_days:number;reference:string};
export const money=(minor:number,currency:string)=>new Intl.NumberFormat(undefined,{style:'currency',currency}).format(minor/100);
