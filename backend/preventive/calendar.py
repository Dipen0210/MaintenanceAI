"""Anchored local calendar intervals with explicit DST and month-end policies."""
import calendar
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo


def occurrence(template,index):
    anchor=datetime.fromisoformat(template['start_local'])
    count=template['interval_count']*index
    unit=template['interval_unit']
    if unit=='months':
        month=anchor.year*12+anchor.month-1+count;year,month=divmod(month,12);month+=1
        wall=anchor.replace(year=year,month=month,day=min(anchor.day,calendar.monthrange(year,month)[1]))
    else:wall=anchor+timedelta(days=count*(7 if unit=='weeks' else 1))
    zone=ZoneInfo(template['timezone'])
    # First occurrence of an ambiguous clock time. A gap moves to the first valid
    # minute on that date, without changing the anchor for subsequent occurrences.
    for shift in range(181):
        candidate=(wall+timedelta(minutes=shift)).replace(tzinfo=zone,fold=0)
        utc=candidate.astimezone(timezone.utc)
        if utc.astimezone(zone).replace(tzinfo=None)==candidate.replace(tzinfo=None):return utc
    raise ValueError('Calendar time falls in an unsupported timezone gap; revise the plan')
