'use client';
import {useEffect,useState} from 'react';
import Link from 'next/link';
import Icon from './Icon';
import {operationsRequest} from '@/lib/operations';
export default function NotificationIndicator(){
 const [count,setCount]=useState<number|null>(null);
 useEffect(()=>{let active=true;const load=()=>operationsRequest<{unread:number}>('/notifications').then(r=>{if(active)setCount(r.unread);}).catch(()=>{if(active)setCount(null);});load();const timer=setInterval(load,10000);window.addEventListener('notifications-updated',load);return()=>{active=false;clearInterval(timer);window.removeEventListener('notifications-updated',load);};},[]);
 return <Link href="/operations/notifications" className="notification-launcher" aria-label={count===null?'Notifications: count unavailable':`Notifications: ${count} unread`}><Icon name="bell" size={20}/><span>{count===null?'—':count>99?'99+':count}</span></Link>;
}
