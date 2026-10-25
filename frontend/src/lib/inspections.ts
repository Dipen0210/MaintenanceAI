export type InspectionStep={asset_id:string;title:string;instructions:string;procedure_ids:string[];procedures?:{id:string;title:string;revision:string;text:string}[];kind:'numeric'|'check';unit:string;lower:number|null;upper:number|null};
export type InspectionContent={title:string;steps:InspectionStep[]};
export type InspectionTemplate={id:string;version:number;enabled:boolean;content:InspectionContent;history:{version:number;actor_id:string;created_at:string;reason:string}[]};
export type InspectionResult={step_index:number;outcome:'measured'|'pass'|'fail'|'skipped';value:number|null;notes:string;abnormal:boolean;issue_id:string|null;actor_id:string;recorded_at:string};
export type InspectionRun={id:string;template_version:number;content:InspectionContent;assignee_id:string;due_at:string;status:string;version:number;overdue:boolean;results:InspectionResult[];events:{version:number;actor_id:string;created_at:string;action:string;detail:{notes?:string;step_index?:number}}[]};
