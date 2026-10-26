'use client';
import Link from 'next/link';
import {usePathname} from 'next/navigation';
export default function InventoryLayout({children}:{children:React.ReactNode}){const path=usePathname();return <><nav className="inventory-tabs" aria-label="Inventory sections">{[['/operations/inventory','Parts'],['/operations/inventory/movements','Stock movements']].map(([href,label])=><Link key={href} href={href} aria-current={path===href?'page':undefined}>{label}</Link>)}</nav>{children}</>;}
