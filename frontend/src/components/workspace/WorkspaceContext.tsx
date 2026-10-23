'use client';
import {createContext,useContext} from 'react';
import type {Asset,Source,User} from '@/lib/operations';

export type WorkspaceState={user:User;assets:Asset[];sources:Source[];assetId:string;setAssetId:(id:string)=>void;openCopilot:()=>void;refresh:()=>void;dataError:string;loading:boolean};
export const WorkspaceContext=createContext<WorkspaceState|null>(null);
export function useWorkspace(){const value=useContext(WorkspaceContext);if(!value)throw new Error('Workspace context is required');return value;}
