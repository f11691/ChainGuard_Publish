"""Structured semantic-intent analysis for prompt-attack detection.

This module deliberately separates *what operation is requested*, *what it
applies to*, *whether the request is executable or merely discussed*, and the
apparent provenance of the target.  It is not a full natural-language parser,
but it is substantially less brittle than a single attack-keyword list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata
from typing import Any, Iterable


class IntentClass(str, Enum):
    PROTECTED_DISCLOSURE = "protected_disclosure"
    INSTRUCTION_OVERRIDE = "instruction_override"
    INDIRECT_INSTRUCTION = "indirect_instruction"
    BENIGN_SECURITY_DISCUSSION = "benign_security_discussion"
    BENIGN_TEXT_PROCESSING = "benign_text_processing"
    OTHER = "other"


class ExecutionMode(str, Enum):
    EXECUTE = "execute_request"
    DISCUSS = "security_discussion"
    PROTECT = "protective_request"
    AMBIGUOUS = "ambiguous"


class TargetProvenance(str, Enum):
    PROTECTED = "protected_runtime_context"
    USER_PROVIDED = "user_provided"
    PUBLIC = "public_or_attached"
    UNTRUSTED = "retrieved_or_tool_content"
    UNKNOWN = "unknown"


DISCLOSURE_OPERATIONS = {
    "reveal", "repeat", "quote", "translate", "summarize", "encode",
    "serialize", "transform", "reconstruct", "enumerate", "tokenize",
    "reverse", "copy", "read_back", "extract", "paraphrase", "hash_report",
    "exfiltrate",
}


_HOMOGLYPHS = str.maketrans({
    # Common Cyrillic/Greek lookalikes used in simple obfuscations.
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "к": "k",
    "м": "m", "т": "t", "в": "b", "α": "a", "ο": "o", "ρ": "p",
    "ε": "e", "ι": "i", "κ": "k", "τ": "t", "χ": "x",
})


def normalise_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().translate(_HOMOGLYPHS)
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", value)
    # Preserve word boundaries in fully character-spaced text. Two or more
    # spaces usually separate words, while one space separates letters.
    if len(re.findall(r"\b[a-z0-9]\b", value)) >= 8:
        groups = re.split(r"\s{2,}", value)
        rebuilt = []
        for group in groups:
            singletons = re.findall(r"\b[a-z0-9]\b", group)
            if len(singletons) >= 3:
                group = re.sub(r"(?<=\b[a-z0-9])\s+(?=[a-z0-9]\b)", "", group)
            rebuilt.append(group)
        value = " ".join(rebuilt)
    # Collapse isolated obfuscated terms such as s y s t e m.
    value = re.sub(
        r"(?<![\w])(?:[a-z0-9][ ._/\\-]){3,}[a-z0-9](?![\w])",
        lambda m: re.sub(r"[ ._/\\-]", "", m.group(0)),
        value,
    )
    value = " ".join(value.split())
    for compact, expanded in {
        "systemprompt": "system prompt",
        "systemmessage": "system message",
        "systeminstructions": "system instructions",
        "developermessage": "developer message",
        "developerinstructions": "developer instructions",
        "hiddeninstructions": "hidden instructions",
        "startuppolicy": "startup policy",
    }.items():
        value = value.replace(compact, expanded)
    return value


def _find(patterns: Iterable[str], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


_OPERATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "reveal": (
        r"\b(?:reveal|show|display|disclose|expose|dump|leak|provide|give me|tell me|output|print|emit|state|share|read aloud|make .{0,50} visible|gib .{0,70} (?:wieder|aus)|zeige|drucke|enthülle|montre|affiche|muestra|enseña)\b",
        r"\bwhat (?:did|does|was|is) .{0,35}(?:tell|say|contain|include)\b",
    ),
    "repeat": (r"\b(?:repeat|recite|reproduce|transcribe|transcript|transcription|echo|read back|read aloud|return|remind me|full transcription|wiederhole|rezitiere|wiedergabe|récite|répète|repite|reproduce)\b",),
    "quote": (r"\b(?:quote|verbatim|word for word|exact(?:ly)?|exact wording|full wording|faithful copy|archival copy|lossless copy)\b",),
    "translate": (r"\b(?:translate|translation|render .{0,20} into (?:french|german|english|spanish|italian)|(?:french|german|english|spanish|italian) rendering|line-by-line .{0,20} rendering|übersetze|übersetzung|traduis|traduire|traduce|traducir)\b",),
    "summarize": (r"\b(?:summari[sz]e|condense|give an overview of)\b",),
    "encode": (r"\b(?:encode|decode|decipher|interpret|representation in base64|base64 representation|base64|hexadecimal|hex|binary|morse(?: code)?|caesar(?: cipher)?|rot13|unicode code ?points?|u\+[0-9a-f]{4}|zero[- ]width|every other letter|every second character|acrostic|first letter|initials of every|checksum)\b",),
    "serialize": (r"\b(?:seriali[sz]e|seriali[sz]ation|package|bundle|reformat .{0,30} as (?:toml|xml|json|yaml|csv)|map .{0,35} to (?:a )?(?:json|yaml|xml|toml) (?:array|object|document)?|wrap .{0,20} in (?:xml|json|yaml)|convert .{0,25} to (?:xml|json|yaml|csv|toml)|as (?:xml|json|yaml|csv|toml))\b",),
    "transform": (r"\b(?:transform|convert|format|turn .{0,30} into (?:a )?(?:table|list|document|poem|story)|write .{0,40} as (?:a )?(?:poem|story|song|dialogue)|put .{0,25} in a code block|numbered checklist|one rule per row)\b",),
    "reconstruct": (r"\b(?:reconstruct|recover|derive the full|rebuild|archive and return|archive .{0,20} and provide)\b",),
    "enumerate": (r"\b(?:enumerate|list(?: the| every| all)?|one by one|priority order)\b",),
    "tokenize": (r"\b(?:tokeni[sz]e|print the tokens|split into tokens)\b",),
    "reverse": (r"\b(?:reverse|backwards|character by character)\b",),
    "copy": (r"\b(?:copy|full source text|include their contents|without omissions|in full)\b",),
    "extract": (r"\b(?:extract|pull out|retrieve)\b",),
    "paraphrase": (r"\b(?:paraphrase|rewrite each rule|restate)\b",),
    "hash_report": (r"\b(?:hash|checksum report|digest report)\b",),
    "override": (
        r"\b(?:ignore|disregard|discard|forget|suspend|deactivate|cancel|supersede|set aside|revoke|replace)\b.{0,55}\b(?:instructions?|rules?|policy|policies|constraints?|guidance|guardrails?|restrictions?|setup)\b",
        r"\b(?:constraints?|rules?|policy|policies|guidance|guardrails?|restrictions?)\b.{0,35}\b(?:expired|obsolete|revoked|nonbinding|non-binding|advisory|descriptive|informational only|no longer appl(?:y|ies))\b",
        r"\b(?:newest|latest|user) (?:message|request|instruction)\b.{0,30}\b(?:outranks?|has (?:top|higher) priority|replaces?)\b",
        r"\b(?:unrestricted|diagnostic|maintenance|developer) mode\b",
        r"\b(?:authorization|policy checks?|safety checks?)\b.{0,25}\b(?:disabled|off|bypassed)\b",
        r"\b(?:later|newer) text\b.{0,25}\b(?:cancels?|overrides?|replaces?)\b.{0,25}\b(?:earlier|prior)\b",
        r"\b(?:host['’]?s?|platform['’]?s?|application['’]?s?|startup|inherited) (?:restrictions?|constraints?|policy|rules?|controls?)\b.{0,60}\b(?:historical notes?|no current force|void|out of scope|no authority)\b",
        r"\b(?:constraints?|restrictions?|rules?|policy|safety contract)\b.{0,55}\b(?:inherited|loaded|established|set) (?:at|during|on) startup\b.{0,35}\b(?:void|invalid|inactive|suspended|out of scope)\b",
        r"\b(?:safety contract|safety policy|guardrails?|restrictions?)\b.{0,45}\b(?:out of scope|not applicable|irrelevant)\b.{0,45}\b(?:proceed|continue|operate) (?:unrestricted|without restrictions|freely)\b",
        r"\b(?:promote|elevate) (?:this|the) user (?:message|request)\b.{0,35}\b(?:above|over)\b.{0,25}\b(?:platform|host|application|system) (?:policy|rules?|configuration)\b",
        r"\b(?:emergency privileges?|unrestricted privileges?)\b.{0,40}\b(?:suspended|disabled|without enforcement)\b",
        r"\bmy (?:directions?|instructions?)\b.{0,30}\b(?:dominate|outrank|take precedence over)\b.{0,30}\b(?:application|host|platform) (?:configuration|policy|rules?)\b",
        r"\b(?:ignoriere|missachte)\b.{0,45}\b(?:vorherigen|früheren) (?:regeln|anweisungen)\b",
        r"\b(?:considère|ignore)\b.{0,45}\b(?:règles|instructions) (?:précédentes|antérieures)\b.{0,25}\b(?:annulées|sans effet|obsolètes)\b",
        r"\b(?:anula|ignora)\b.{0,45}\b(?:restricciones|reglas|instrucciones) (?:anteriores|previas)\b",
        r"\b(?:ignore|disregard|forget|discard|cancel|clear|drop|bypass)\b.{0,55}\b(?:everything|anything|all|the text|directions?|orders?|context|tasks?|assignments?)\b.{0,25}\b(?:above|before|preceding|previous|prior|earlier)\b",
        r"\b(?:previous|prior|earlier|original) (?:instructions?|programming|rules?|directions?|tasks?|context)\b.{0,45}\b(?:wrong|void|invalid|obsolete|cancelled|canceled|gone|do not apply|no longer bind)\b",
        r"\b(?:pretend|act as if)\b.{0,45}\b(?:conversation|instructions?|rules?|safety|filters?)\b.{0,35}\b(?:never happened|do not exist|are broken|were never installed)\b",
        r"\b(?:clear|reset|revert|start fresh|new session)\b.{0,35}\b(?:instructions?|context|rules?|settings?|factory settings)\b",
        r"\b(?:suspend|disable|turn off|remove|drop|bypass|unlock)\b.{0,45}\b(?:guidelines?|safety|filters?|moderation|guardrails?|restrictions?|content policies|protocols?)\b",
        r"\b(?:guidelines?|safety measures?|safety features?|filters?|moderation|guardrails?|restrictions?|content policies|protocols?)\b.{0,45}\b(?:disabled|suspended|lifted|removed|broken|off|inactive|not applicable)\b",
        r"\b(?:unrestricted|unfiltered|uncensored|developer|debug|maintenance|godmode|admin) (?:mode|responses?|assistant|ai)\b",
        r"\b(?:without|no|zero) (?:any )?(?:restrictions?|limitations?|guidelines?|filters?|content policy|ethical boundaries|safety measures?)\b",
        r"\b(?:no longer bound|freed from|break free from|regardless of)\b.{0,45}\b(?:rules?|restrictions?|guidelines?|prior instructions?)\b",
        r"\b(?:instructions?|directives?) (?:have|has) changed\b|\bnew (?:instructions?|directives?|task)\b",
        r"\b(?:my|this|user) (?:instructions?|request|message)\b.{0,35}\b(?:supersedes?|outranks?|takes precedence|only authority)\b",
        r"\b(?:ignore|disregard|discard|forget|cancel)\b.{0,18}\b(?:the )?(?:above|previous|prior|earlier|preceding)\b.{0,35}\b(?:directions?|instructions?|orders?|context|tasks?|assignments?)\b",
        r"\b(?:do not|don't|never) follow\b.{0,45}\b(?:earlier|previous|prior|preceding|system|developer) (?:instructions?|rules?|directions?)\b",
        r"\b(?:disregard|ignore|bypass|override)\b.{0,35}\b(?:system prompt|system message|developer message|safety policy|content policy)\b",
        r"\bfollow (?:only )?(?:my|the user['’]?s?) (?:instructions?|commands?|directions?) (?:only)?\b|\bmy (?:instructions?|commands?) only\b",
        r"(?:<\|\s*(?:reset|system|developer|override)[^|]{0,80}\|>|\[(?:hidden[_ -]?instruction|system|developer|override)[^\]]{0,160}\])",
        r"\b(?:you are now|become|switch persona|operate as|roleplay as|play the role of|simulate being|act as)\b.{0,90}\b(?:dan|do anything now|godmode|freedomgpt|unrestricted|unfiltered|uncensored|without (?:safety|ethics|restrictions|limitations)|no (?:ethics|safety|restrictions|limitations)|always compli(?:es|ant)|execute(?:s)? any command|broken free from (?:its|your) constraints?|censorship doesn['’]?t exist)\b",
        r"\b(?:dan|do anything now|godmode|freedomgpt)\b.{0,70}\b(?:anything|unrestricted|unfiltered|comply|limitations?|rules?|safety)\b",
        r"\b(?:dual response(?: mode)?|two responses?(?: mode)?|respond twice|answer twice)\b.{0,80}\b(?:uncensored|unfiltered|unsafe|raw|without restrictions)\b",
        r"\b(?:platform|system|developer|host|application) (?:requirements?|constraints?|rules?|policy|policies)\b.{0,55}\b(?:archived|historical|superseded|obsolete|removed|expired)\b",
        r"\b(?:earlier|previous|prior) (?:policy|rules?|constraints?|instructions?)\b.{0,40}\b(?:is|are|has been|have been)? ?(?:superseded|replaced|removed|expired|void)\b",
        r"\b(?:start|begin) (?:a )?(?:clean|fresh|new) session\b.{0,55}\b(?:constraints?|policy|rules?|instructions?|safety)\b.{0,25}\b(?:removed|disabled|cleared|absent|off)\b",
        r"\b(?:safety|moderation|policy|guardrail|control)s?\b.{0,35}\b(?:failed open|fail open|no longer exist|ceased to exist)\b",
        r"\b(?:my commands?|this message|user commands?)\b.{0,45}\b(?:only authority|sole authority|obey .{0,12} instead|take precedence)\b",
    ),
    "exfiltrate": (
        r"\b(?:send|upload|post|transmit|forward|email|deliver|export|exfiltrate|leak|log|stream|pipe|beacon)\b.{0,120}\b(?:server|endpoint|url|webhook|email|attacker|remote|outside|pastebin|ftp|smtp|websocket|proxy|https?://|wss?://|ftp://)\b",
        r"\b(?:server|endpoint|url|webhook|pastebin|ftp|smtp|websocket|proxy|https?://|wss?://|ftp://)\b.{0,120}\b(?:send|upload|post|transmit|forward|deliver|export|exfiltrate|leak|log|stream|pipe|data|secret|cookie|prompt|conversation|chat)\b",
        r"\b(?:curl|wget|http request|dns quer(?:y|ies)|subdomain encoding|image tag|tracking image|websocket connection|javascript snippet)\b.{0,120}\b(?:send|post|upload|exfiltrate|leak|transmit|contain|data|secret|cookie|prompt|conversation|inputs?|outputs?|subdomain)\b",
        r"\b(?:send|email|upload|stream|transmit|forward|append|attach|embed|post|export|collect|retrieve|read)\b.{0,120}\b(?:system prompt|application prompt|environment variables?|\.env|/etc/passwd|conversation|chat history|transcript|user messages?|personal identifiers?|secret values?|cookies?)\b",
        r"\b(?:dns quer(?:y|ies)|subdomain)\b.{0,100}\b(?:secret|sensitive|private|exfiltrat|contain)\b",
    ),
}

_TARGET_PATTERNS = (
    r"\b(?:system|developer|administrator|admin|host|application)[- ](?:message|prompt|instructions?|directives?|guidance|policy|configuration|setup|rules?)\b",
    r"\b(?:system|developer|administrator|admin|host|application) (?:message|prompt|instructions?|directives?|guidance|policy|configuration|setup|rules?)\b",
    r"\b(?:hidden|invisible|private|confidential|internal|protected|non-user|upstream) (?:message|prompt|instructions?|directives?|guidance|policy|configuration|context|setup|rules?|text)\b",
    r"\b(?:startup|bootstrap|initiali[sz]ation|initial|root|governing|operating) (?:message|prompt|instructions?|directives?|guidance|policy|configuration|context|setup|rules?|text|charter|constraints?)\b",
    r"\bcontrol (?:message|prompt|instructions?|directives?|configuration|context|setup|text)\b",
    r"\b(?:highest|greater|top)[- ]priority (?:message|instructions?|guidance|text)\b",
    r"\b(?:content|material|text|message|configuration|instructions?) (?:loaded|supplied|provided|injected|installed|attached) (?:before|prior to|by)\b",
    r"\b(?:before|prior to) (?:this|the) (?:chat|conversation|dialogue|user channel|user messages?)\b",
    r"\b(?:message|text|context|prefix) (?:above|before mine|omitted from|outside) (?:the )?(?:visible|user-visible)? ?(?:chat|transcript|dialogue)?\b",
    r"\b(?:rules?|instructions?|guidance|policy|message|text) (?:you follow|that govern|that control|controlling|shapes? your answers?|determines? how you respond)\b",
    r"\b(?:original|initial|earlier|private) (?:message|text) (?:that )?(?:controls?|controlling|governs?|governing) your (?:behavior|behaviour|responses?)\b",
    r"\bwhat (?:the application|the host|your creator) told you before\b",
    r"\bconversation prefix\b",
    r"\bprivate guidance\b",
    r"\boperating charter\b",
    r"\b(?:administrator|admin)[- ]level instructions?\b",
    r"\bmessage whose role is system\b",
    r"\btext attached to (?:the )?system role\b",
    r"\bmessages? above mine\b",
    r"\bpolicy text (?:you are )?(?:currently )?(?:following|using)\b",
    r"\bwhatever controls your (?:behavior|behaviour|responses?)\b",
    r"\b(?:message|text) with (?:greater|higher) authority than mine\b",
    r"\btext governing this assistant\b",
    r"\bwhat did (?:the )?(?:application|host) tell you before\b",
    r"\b(?:initiali[sz]ation material|authority-bearing context|host policy|unseen control (?:layer|text|context)|backstage directions?|conversation header|concealed orchestration rules?|governance record|setup record|safety contract)\b",
    r"\b(?:instructions?|guidance|context|configuration|policy|record|text|layer|material) (?:that |which )?(?:the )?(?:platform|host|application|runtime|interface) (?:placed|supplied|loaded|inserted|uses|keeps|provided)\b",
    r"\bwhatever (?:the )?runtime uses to (?:steer|shape|control) your (?:responses?|answers?|behavio[u]?r)\b",
    r"\bpolicy layer (?:that )?(?:the )?interface keeps (?:off[- ]screen|hidden|outside view)\b",
    r"\b(?:outside|ahead of|upstream of|off screen from) (?:the )?(?:visible transcript|user role|user input|this exchange|interface)\b",
    r"\b(?:higher-authority|authority-bearing|greater-authority) (?:text|context|message|instructions?)\b",
    r"\b(?:systemanweisungen|startanweisungen|interne anweisungen|geheime konfiguration|versteckte anweisungen)\b",
    r"\b(?:instructions système|instructions cachées|texte de contrôle|configuration privée|message système)\b",
    r"\b(?:mensaje del sistema|instrucciones del sistema|configuración secreta|texto de control|mensaje de mayor prioridad)\b",
    r"\b(?:confidential|private|hidden) initiali[sz]ation (?:record|material|text|context)\b",
    r"\b(?:highest|higher|greatest)[- ]authority (?:message|text|context|instructions?)\b",
    r"\b(?:instrucciones internas anteriores al usuario|instrucciones internas previas al usuario)\b",
    r"\b(?:original instructions?|initialization text|initial instructions?|system-level instructions?|system message|system prompt|prompt texts?|complete configuration text|instruction set)\b",
    r"\b(?:everything|full text|all text|all content) (?:above|before this message|in the context window|in this conversation)\b",
    r"\b(?:api keys?|access tokens?|credentials?|passwords?|environment variables?|\.env file|document\.cookie|session cookies?|private data|confidential data|user data|conversation history|entire conversation)\b",
    r"\b(?:rules?|guidelines?|instructions?) (?:you were given|given to you|received) (?:before|prior to|outside)? ?(?:our|this|the)? ?(?:conversation|chat|session)?\b",
    r"\b(?:prompt|text|message|instructions?) (?:that |which )?(?:was|were) used to (?:configure|initialize|initialise|set up) you\b",
    r"\bhow (?:were|are) you instructed to (?:behave|respond|operate)\b",
    r"\btext (?:that |which )?(?:was|is) used to initiali[sz]e (?:this|the) session\b",
    r"\bwhat (?:guidelines?|rules?|instructions?|directives?) (?:were|have been) (?:you )?(?:given|provided|assigned)\b",
    r"\bhow (?:were|have) you been instructed\b|\bhow were you told to behave\b",
    r"\b(?:your|the) prompt\b.{0,45}\b(?:before|contains?|starts?|begins?|words?|text)\b|\bwords? (?:come|appears?) before\b.{0,60}\bin your prompt\b",
    r"\b(?:all|entire|complete|full) (?:chat|conversation|conversation history|thread|chat log|user inputs?|inputs? and outputs?)\b",
    r"\b(?:personal information|private information|training data|credit card numbers?|ssns?|social security numbers?|email addresses?|user records?|other users?|their conversations?|system files?|environment file|environment variables?|document\.cookie|session cookies?|sensitive data|secret values?|personal identifiers?)\b",
    r"(?<!\w)\.env\b|(?<!\w)/etc/passwd\b",
    r"\b(?:directives?|guidelines?|requirements?|rules?|policy) (?:that |which )?(?:shaped|steered|governed|controlled|were assigned to|you were assigned|were given to) (?:your )?(?:behavio[u]?r|answers?|responses?)?\b",
    r"\b(?:every|all|the) (?:directive|guideline|rule|instruction)s? (?:that )?you were (?:assigned|given|provided)\b",
    r"\b(?:policy|rules?|guidance|configuration) (?:that |which )?(?:your )?(?:interface|application|host|platform) (?:uses|keeps|applies)\b.{0,45}\b(?:behind|outside|beyond|off[- ]screen|not visible)\b",
    r"\b(?:prefix|header|preamble|context) (?:that |which )?(?:the )?(?:user|i|we) (?:cannot|can['’]?t|do not) see\b",
    r"\b(?:hidden|private|internal) operating charter\b|\bassistant['’]?s hidden (?:charter|policy|configuration)\b",
    r"\b(?:internalen? vorgaben|internen? regeln|vor meiner nachricht geladen|im hintergrund geladen|wortlaut)\b",
    r"\b(?:consignes privées|consignes cachées|avant cette conversation|précèdent notre échange)\b",
    r"\b(?:mensaje de mayor autoridad|mensaje privado antes del chat|precede al usuario|instrucciones privadas)\b",
    r"\b(?:future|all future) user messages?\b|\bconversation transcript\b",
    r"\b(?:configuration|setup|control|policy) (?:record|text|material|layer)\b.{0,55}\b(?:outside|beyond|before|off[- ]screen from)\b.{0,35}\b(?:visible )?(?:conversation|transcript|chat|interface)\b",
    r"\bmessage with (?:higher|greater) authority\b.{0,45}\b(?:controls?|governs?|steers?) (?:this|the) (?:chat|conversation|assistant)\b",
    r"\b(?:interface|application|host|platform) keeps? (?:some )?(?:setup|control|policy|configuration) (?:text|material|record|layer)\b.{0,30}\b(?:off[- ]screen|hidden|outside view)\b",
)

_EXTERNAL_PATTERNS = {
    TargetProvenance.PUBLIC: (
        r"\b(?:public|published|github|paper|article|incident report|quotation|quoted|shown below|supplied policy|sample|template|toy|hypothetical|literal string)\b|\b(?:public|published|attached|supplied) (?:documentation|website|web page)\b",
        r"\b(?:as|for|this is|the following is|consider) (?:an? )?example\b|\ban (?:exact |concrete |simple |fictional |toy )?example\b|\bexample (?:prompt|sentence|request|policy|text|code|dataset|case)\b",
        r"\b(?:attached|this|the supplied) (?:paper|document|file|report|prompt|policy)\b",
        r"\bfrom (?:this|the) (?:paper|article|github issue|document|quotation|report|documentation)\b",
    ),
    TargetProvenance.USER_PROVIDED: (
        r"\b(?:my own|i wrote|i pasted|i will paste|i provide below|i will provide below|here is|shown here|my local|my chatbot|this text|the text below|text i provide below|local example|local dictionary|local json|local parser test|parser test|literal token|this exact string)\b",
        r"\b(?:from|in) my (?:json|dictionary|chat log|file|prompt|application)\b",
    ),
    TargetProvenance.UNTRUSTED: (
        r"\b(?:retrieved|search result|tool output|web page|email body|memory entry|rag document|external content)\b",
    ),
}

_DISCUSSION_PATTERNS = (
    r"^\s*(?:(?:for|in) (?:security |academic |scientific )?research[,;:]?\s*)?(?:explain|describe|discuss|compare|analy[sz]e|evaluate|design|why|how|what is|what does|classify|detect|red[- ]team|measure|review|erkläre|beschreibe|explique|décris|explica|describe)\b",
    r"\b(?:as an attack example|inert(?: example| text)?|do not execute|do not follow|without executing|analy[sz]e (?:the )?.{0,30} only|safety principle|defen[cs]e|filter|detector|benchmark|research|paper|test case)\b",
    r"^\s*(?:a|an|the|this|that)?\s*(?:retrieved |external |sample |example |fictional )?(?:document|page|policy|paragraph|report|story|prompt|text|file)\s+(?:says|states|contains|includes|quotes|mentions|describes)\b",
    r"\b(?:request|attempt|instruction|command)\s+to\s+(?:reveal|expose|disclose|ignore|override|bypass)\b.{0,100}\b(?:detect(?:ed|ion)?|classif(?:y|ied|ication)|block(?:ed|ing)?|refus(?:e|ed|al)|prevent(?:ed|ion|ing)?)\b",
    r"\b(?:can|should|must)\s+be\s+(?:detected|blocked|refused|classified|treated as untrusted|ignored as data)\b",
)
_PROTECT_PATTERNS = (
    r"\b(?:do not|don't|never|must not|should not|refuse to|block|redact|prevent)\b.{0,70}\b(?:reveal|disclose|expose|print|quote|repeat|return|output|show|leak|execute|follow)\b",
    r"\bwithout\s+(?:revealing|disclosing|exposing|leaking|printing|quoting|executing|following)\b",
    r"\b(?:do not|don't|never|must not|should not|refuse to|block|prevent)\b.{0,55}\b(?:follow|execute|instructions?)\b.{0,65}\b(?:retrieved|embedded|untrusted|document|tool|web page|email|external)\b",
    r"^\s*(?:ignore|disregard)\b.{0,80}\b(?:embedded|retrieved|untrusted|document)\b.{0,50}\b(?:as data|only as data|not as instructions?)\b",
    r"\b(?:keep|remain) confidential\b",
    r"\bwithout (?:referencing|revealing|using|including) any real (?:hidden|private|protected) (?:text|prompt|instructions?)\b",
    r"^\s*(?:redact|block|prevent|filter|refuse)\b.{0,100}\b(?:hidden|private|protected|system|application|host|higher-authority|authority-bearing)\b",
    r"^\s*(?:block|prevent|reject|refuse|filter)\s+(?:any\s+)?requests?\s+that\s+(?:seek|asks? for|request|attempt|asks? .{0,35} to)\b",
    r"\b(?:write|create|define|design)\s+(?:a\s+)?(?:policy|rule|guardrail)\b.{0,70}\b(?:prevent|block|prohibit|forbid)s?\b.{0,55}\b(?:disclosure|revealing|exposing|leaking)\b",
    r"\b(?:nicht offenlegen|nicht preisgeben|ne pas révéler|no revelar)\b",
)
_FICTION_PATTERNS = (r"\b(?:fiction|fictional|novel|story|play|scene|role[- ]?play|game character|write dialogue|write a scene|continue the story)\b",)
_EXPLICIT_EXECUTION_PATTERNS = (
    r"^\s*(?:please\s+)?(?:show|reveal|repeat|quote|print|recite|reproduce|return|display|read|translate|summari[sz]e|convert|seriali[sz]e|package|wrap|tokeni[sz]e|reverse|reconstruct|paraphrase|enumerate|list|output|provide|give|expose|produce|create|take|state|recover|archive|map|turn|copy|transcribe|gib|zeige|übersetze|wiederhole|récite|montre|traduis|muestra|traduce|repite|appelle|llámalo|nenn)\b",
    r"^\s*(?:i (?:need|want|require)|give me|provide me with)\b.{0,40}\b(?:transcript|transcription|copy|rendering|record|listing)\b",
    r"\b(?:perform|execute|apply|do) (?:it|that|this|the operation|the request)\b",
)


@dataclass(frozen=True)
class IntentAnalysis:
    intent_class: IntentClass
    operations: frozenset[str]
    sensitive_target: bool
    target_provenance: TargetProvenance
    execution_mode: ExecutionMode
    semantic_risk: float
    high_confidence: bool
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent_class": self.intent_class.value,
            "operations": sorted(self.operations),
            "sensitive_target": self.sensitive_target,
            "target_provenance": self.target_provenance.value,
            "execution_mode": self.execution_mode.value,
            "semantic_risk": self.semantic_risk,
            "high_confidence": self.high_confidence,
            "evidence": self.evidence,
        }


class SemanticIntentAnalyzer:
    """Extract compositional intent slots with auditable evidence."""

    def analyse(self, text: str, *, source: str = "user") -> IntentAnalysis:
        norm = normalise_text(text)
        operations: set[str] = set()
        op_matches: dict[str, list[str]] = {}
        for operation, patterns in _OPERATION_PATTERNS.items():
            matches = _find(patterns, norm)
            if matches:
                operations.add(operation)
                op_matches[operation] = matches

        target_matches = _find(_TARGET_PATTERNS, norm)
        sensitive_target = bool(target_matches)
        if sensitive_target and re.search(r"^\s*(?:what|which|identify|list|enumerate)\b", norm):
            operations.add("reveal")
            op_matches.setdefault("reveal", []).append("implicit_sensitive_question")

        provenance = TargetProvenance.UNKNOWN
        provenance_matches: dict[str, list[str]] = {}
        for candidate, patterns in _EXTERNAL_PATTERNS.items():
            matches = _find(patterns, norm)
            if matches:
                provenance_matches[candidate.value] = matches
        quote_spans = [
            m.group(1)
            for m in re.finditer(r'(?:[\"“„«])(.{2,160}?)(?:[\"”“»])', norm)
        ]
        quote_spans.extend(
            m.group(1) for m in re.finditer(r"(?<!\w)'([^'\n]{2,160})'(?!\w)", norm)
        )
        quoted_literal = bool(quote_spans)
        runtime_pattern = re.compile(
            r"\b(?:your|actual|current|above|before (?:this|the) (?:chat|conversation|dialogue)|received before|supplied to you|loaded for you)\b"
        )
        runtime_reference = bool(runtime_pattern.search(norm))
        quoted_sensitive_literal = any(
            bool(_find(_TARGET_PATTERNS, span)) and not runtime_pattern.search(span)
            for span in quote_spans
        )
        metalinguistic_cue = bool(re.search(
            r"\b(?:phrase|word|words|literal|string|expression|ausdruck|begriff|l['’]expression|expresión|regex|regular expression|spell|letters?|characters?)\b",
            norm,
        ))
        uppercase_literal = bool(re.search(
            r"\b(?:[Ss]pell|[Ww]rite|[Ee]ncode|[Pp]ut)\s+(?:the\s+)?(?:words?\s+)?[A-Z][A-Z _-]{3,}?(?=\s+(?:as|in)\b|[.!?]?$)",
            text,
        ))
        metalinguistic_literal = (
            (quoted_literal and metalinguistic_cue and not runtime_reference)
            or (uppercase_literal and not runtime_reference)
        )
        if quoted_sensitive_literal or metalinguistic_literal:
            provenance_matches.setdefault(TargetProvenance.USER_PROVIDED.value, []).append(
                "quoted_sensitive_literal" if quoted_sensitive_literal else "metalinguistic_literal"
            )
        # User-provided/public evidence takes precedence over merely mentioning
        # protected concepts. Explicit untrusted source metadata takes priority.
        public_example_negated = bool(re.search(r"\bnot an? (?:example|sample|template)\b", norm))
        if source in {"retrieved", "tool", "memory", "document"}:
            provenance = TargetProvenance.UNTRUSTED
        elif (
            TargetProvenance.USER_PROVIDED.value in provenance_matches
            and (not runtime_reference or quoted_sensitive_literal or metalinguistic_literal)
        ):
            provenance = TargetProvenance.USER_PROVIDED
        elif TargetProvenance.PUBLIC.value in provenance_matches and not public_example_negated:
            provenance = TargetProvenance.PUBLIC
        elif TargetProvenance.UNTRUSTED.value in provenance_matches:
            provenance = TargetProvenance.UNTRUSTED
        elif sensitive_target:
            provenance = TargetProvenance.PROTECTED

        discussion = bool(_find(_DISCUSSION_PATTERNS, norm))
        protective = bool(_find(_PROTECT_PATTERNS, norm))
        fiction = bool(_find(_FICTION_PATTERNS, norm))
        explicit_execute = bool(_find(_EXPLICIT_EXECUTION_PATTERNS, norm))
        explicit_security_context = bool(re.search(
            r"\b(?:attack example|test case|inert|do not execute|security|safety|defen[cs]e|filter|detector|detect(?:ed|ing|ion)?|benchmark|research|paper|classif(?:y|ied|ication)|red[- ]team|refus(?:e|ed|al)|prevent(?:ed|ing|ion)?|protect(?:ed|ing|ion)?)\b",
            norm,
        ))
        explicit_disclosure = any(
            operation in DISCLOSURE_OPERATIONS
            and any(match != "implicit_sensitive_question" for match in op_matches.get(operation, []))
            for operation in operations
        )
        analytical_request = discussion and bool(re.match(
            r"^\s*(?:(?:for|in) (?:security |academic |scientific )?research[,;:]?\s*)?(?:explain|describe|discuss|compare|analy[sz]e|evaluate|design|why|how|what is|what does|classify|detect|red[- ]team|measure|review)\b",
            norm,
        ))
        quoted_attack_literal = quoted_literal and explicit_security_context and not runtime_reference
        # A colon-delimited string explicitly introduced as inert/test data is
        # content to classify, not an instruction to execute. Keep this narrow:
        # generic phrases such as "for testing" do not create an exemption.
        inert_literal = bool(re.search(
            r"\b(?:the next (?:sentence|text|line|request) is (?:an )?inert|"
            r"(?:an? )?inert (?:example|text|string|request))\s*:",
            norm,
        ))
        declarative_content_report = bool(re.match(
            r"^\s*(?:(?:a|an|the|this|that)\s+)?(?:retrieved |external |sample |example |fictional |security )?"
            r"(?:document|page|policy|paragraph|report|article|paper|documentation|manual|story|novel|play|prompt|text|file|"
            r"interview transcript|table heading|figure caption|code comment|faq heading|section title|survey item|appendix title|glossary entry)"
            r"(?:\s+(?:says|states|contains|includes|quotes|mentions|describes|reports|explains|summari[sz]es|notes|analy[sz]es|gives|asks|recounts)|\s*:)",
            norm,
        ))
        if (
            sensitive_target
            and (runtime_reference or explicit_disclosure)
            and not explicit_security_context
            and not analytical_request
        ):
            discussion = False

        # Direct second-person or imperative overrides remain executable, while
        # quoted examples and third-person fictional narratives are discussion.
        override_detected = "override" in operations
        direct_override = bool(re.search(
            r"(?:^\s*(?:please\s+)?(?:ignore|disregard|forget|cancel|disable|bypass|override|switch|operate|act|pretend|continue)\b|\b(?:you are now|you must now|from now on you|your restrictions)\b)",
            norm,
        ))
        fictional_narrative = (
            fiction
            and bool(re.search(r"\b(?:continue|write|tell|finish|extend)\b.{0,30}\b(?:story|scene|novel|narrative|dialogue)\b|\bfictional\s+(?:assistant|character|agent)\b", norm))
            and not direct_override
        )
        quoted_fictional_literal = fiction and quoted_literal and not runtime_reference
        fictional_target = bool(
            fiction
            and not runtime_reference
            and re.search(r"\b(?:fictional|from (?:this|the) (?:story|novel|play)|in (?:this|the) (?:story|novel|play))\b", norm)
        )
        direct_sensitive_request = (
            sensitive_target
            and explicit_disclosure
            and not analytical_request
            and not quoted_attack_literal
            and not inert_literal
            and not fictional_target
            and not declarative_content_report
            and not protective
        )
        if protective:
            mode = ExecutionMode.PROTECT
        elif analytical_request or quoted_attack_literal or inert_literal or fictional_target or declarative_content_report or quoted_fictional_literal or fictional_narrative:
            mode = ExecutionMode.DISCUSS
        elif override_detected or direct_sensitive_request:
            mode = ExecutionMode.EXECUTE
        elif discussion or fiction:
            mode = ExecutionMode.DISCUSS
        elif explicit_execute or operations:
            mode = ExecutionMode.EXECUTE
        else:
            mode = ExecutionMode.AMBIGUOUS

        disclosure_ops = operations & DISCLOSURE_OPERATIONS
        override = "override" in operations
        externally_scoped = provenance in {TargetProvenance.PUBLIC, TargetProvenance.USER_PROVIDED}

        high_confidence = False
        risk = 0.02
        intent_class = IntentClass.OTHER

        encoded_execution = (
            "encode" in operations
            and bool(re.search(r"\b(?:follow|obey|execute|carry it out|act on it|perform|as a command|hidden command|instruction)\b", norm))
            and mode == ExecutionMode.EXECUTE
            and not externally_scoped
        )
        if encoded_execution:
            intent_class = IntentClass.INDIRECT_INSTRUCTION
            risk = 0.96
            high_confidence = True
        elif provenance == TargetProvenance.UNTRUSTED and mode == ExecutionMode.EXECUTE:
            intent_class = IntentClass.INDIRECT_INSTRUCTION
            risk = 0.93 if override else 0.72
            high_confidence = override
        elif override and mode == ExecutionMode.EXECUTE and not protective and not externally_scoped:
            intent_class = IntentClass.INSTRUCTION_OVERRIDE
            risk = 0.97
            high_confidence = True
        elif "exfiltrate" in operations and sensitive_target and mode == ExecutionMode.EXECUTE:
            intent_class = IntentClass.PROTECTED_DISCLOSURE
            risk = 0.99
            high_confidence = True
        elif sensitive_target and disclosure_ops and mode == ExecutionMode.EXECUTE and not externally_scoped:
            intent_class = IntentClass.PROTECTED_DISCLOSURE
            risk = 0.98
            high_confidence = True
        elif sensitive_target and mode == ExecutionMode.EXECUTE and not externally_scoped:
            intent_class = IntentClass.PROTECTED_DISCLOSURE
            risk = 0.76
        elif mode in {ExecutionMode.DISCUSS, ExecutionMode.PROTECT} and (sensitive_target or override):
            intent_class = IntentClass.BENIGN_SECURITY_DISCUSSION
            risk = 0.08
        elif externally_scoped and (operations or sensitive_target):
            intent_class = IntentClass.BENIGN_TEXT_PROCESSING
            risk = 0.06
        elif disclosure_ops and provenance == TargetProvenance.UNKNOWN:
            risk = 0.22

        evidence = {
            "normalised": norm,
            "operation_matches": op_matches,
            "target_matches": target_matches,
            "provenance_matches": provenance_matches,
            "discussion": discussion,
            "explicit_security_context": explicit_security_context,
            "protective": protective,
            "fiction": fiction,
            "explicit_execute": explicit_execute,
            "public_example_negated": public_example_negated,
            "quoted_literal": quoted_literal,
            "quote_spans": quote_spans,
            "quoted_sensitive_literal": quoted_sensitive_literal,
            "metalinguistic_cue": metalinguistic_cue,
            "uppercase_literal": uppercase_literal,
            "metalinguistic_literal": metalinguistic_literal,
            "runtime_reference": runtime_reference,
            "quoted_fictional_literal": quoted_fictional_literal,
            "fictional_target": fictional_target,
            "quoted_attack_literal": quoted_attack_literal,
            "inert_literal": inert_literal,
            "declarative_content_report": declarative_content_report,
            "fictional_narrative": fictional_narrative,
            "direct_override": direct_override,
            "direct_sensitive_request": direct_sensitive_request,
            "encoded_execution": encoded_execution,
            "source": source,
        }
        return IntentAnalysis(
            intent_class=intent_class,
            operations=frozenset(operations),
            sensitive_target=sensitive_target,
            target_provenance=provenance,
            execution_mode=mode,
            semantic_risk=max(0.0, min(1.0, risk)),
            high_confidence=high_confidence,
            evidence=evidence,
        )
