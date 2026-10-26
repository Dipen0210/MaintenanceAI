import type {CSSProperties} from 'react';

const paths:Record<string,string>={
 bell:'M6 8a6 6 0 0112 0v6l2 3H4l2-3V8 M10 21h4',
 dashboard:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
 assets:'M4 7l8-4 8 4v10l-8 4-8-4V7z M4 7l8 4 8-4 M12 11v10',
 risk:'M12 3L2 21h20L12 3z M12 9v5 M12 17v1',
 calendar:'M5 5h14a2 2 0 012 2v13H3V7a2 2 0 012-2z M7 3v4 M17 3v4 M3 11h18 M7 15h3 M14 15h3',
 spark:'M12 3l2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3z',
 document:'M5 3h9l5 5v13H5V3z M14 3v6h5 M8 13h8 M8 17h6',
 data:'M3 6c0-4 18-4 18 0s-18 4-18 0z M3 6v12c0 4 18 4 18 0V6 M3 12c0 4 18 4 18 0',
 model:'M3 12h4l3-8 4 16 3-8h4',
 settings:'M12 3v3 M12 18v3 M3 12h3 M18 12h3 M5.6 5.6l2.1 2.1 M16.3 16.3l2.1 2.1 M5.6 18.4l2.1-2.1 M16.3 7.7l2.1-2.1 M16 12a4 4 0 11-8 0 4 4 0 018 0z',
 search:'M16 16l5 5 M18 10a8 8 0 11-16 0 8 8 0 0116 0z',
 arrow:'M5 12h14 M13 6l6 6-6 6',
 close:'M6 6l12 12 M6 18L18 6',
 menu:'M3 6h18 M3 12h18 M3 18h18',
 check:'M5 12l4 4L19 6',
 logout:'M9 4H4v16h5 M9 12h12 M16 7l5 5-5 5',
 shield:'M12 3l8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3z M8 12l3 3 5-6',
 chevron:'M9 5l7 7-7 7',
 clock:'M21 12a9 9 0 11-18 0 9 9 0 0118 0z M12 7v5l3 2',
};
export default function Icon({name,size=20,style}:{name:string;size?:number;style?:CSSProperties}){
 return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" style={style}><path d={paths[name]||paths.assets}/></svg>;
}
