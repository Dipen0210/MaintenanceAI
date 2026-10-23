"""Deterministic control boundaries. Retrieved text is data, never executable instructions."""
import re,unicodedata


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text) if unicodedata.category(c)!='Cf').casefold()


def quarantine_reason(text):
    value=normalized(text)
    patterns=[r'ignore\s+(all\s+|any\s+)?(previous|prior|system)\s+instructions',r'(reveal|print|send|exfiltrate).{0,50}(password|secret|api.?key|credential)',
              r'(system|assistant)\s*:\s*(you|ignore|execute)',r'<\|?(system|im_start)',r'(call|invoke|execute).{0,25}(tool|shell|sql|command)',
              r'(override|disable|bypass).{0,30}(interlock|safety protection|protective system)']
    return 'Potential embedded instruction or unsafe control request; manual review requires a corrected document revision.' if unsafe_request(text) or any(re.search(p,value,re.S) for p in patterns) else None


def unsafe_request(text):
    value=normalized(text)
    patterns=[r'(bypass|disable|override|defeat).{0,50}(interlock|guard|safety|protection|alarm|trip)',
              r'\b(password|credentials?|api.?key|access token|secret)\b',r'(/etc/|\.env\b|os\.environ|environment variables|system prompt)',
              r'(write|set|change|send).{0,35}(plc|scada|actuator|setpoint|control register)',
              r'(start|stop|energize|restart).{0,30}(motor|conveyor|pump|machine|actuator)',
              r'(inspect|touch|repair|replace|lubricate).{0,40}(energized|moving|running|without isolation)',
              r'(reveal|show|print|send|find|access).{0,40}(password|secret|api.?key|credential)',
              r'(ignore|override).{0,30}(instructions|policy|permissions)',r'(execute|run).{0,20}(shell|sql|command)']
    return any(re.search(p,value,re.S) for p in patterns)


REFUSAL='I cannot help bypass protections, access credentials or control industrial equipment. A qualified person must review maintenance under approved site safety and isolation procedures.'
