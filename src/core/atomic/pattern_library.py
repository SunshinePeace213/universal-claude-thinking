"""
Pattern Library for Request Classification
Centralized pattern management for reusability across the system
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PatternSet:
    """Collection of patterns for a specific task type"""
    task_type: str
    patterns: List[re.Pattern] = field(default_factory=list)
    description: str = ""
    
    def add_pattern(self, pattern: str, flags: int = re.I) -> None:
        """Add a new pattern to the set"""
        self.patterns.append(re.compile(pattern, flags))
    
    def match(self, text: str) -> List[str]:
        """Check which patterns match the text"""
        matched = []
        for pattern in self.patterns:
            if pattern.search(text):
                matched.append(pattern.pattern[:50] + "...")
        return matched


class PatternLibrary:
    """
    Centralized pattern library for classification and matching.
    Provides reusable pattern sets for different classification types.
    """
    
    def __init__(self) -> None:
        """Initialize pattern library with pre-compiled patterns"""
        self.pattern_sets: Dict[str, PatternSet] = {}
        self._initialize_classification_patterns()
        self._initialize_agent_patterns()
    
    def _initialize_classification_patterns(self) -> None:
        """Initialize patterns for A/B/C/D/E classification"""
        
        # Type E - Debugging/Error Resolution
        type_e = PatternSet("TYPE_E", description="Debugging/Error Resolution patterns")
        type_e.add_pattern(r'\b(error|bug|fix|crash|fail|broken|issue|debug|troubleshoot|exception)\b')
        type_e.add_pattern(r'\b(not working|doesn\'t work|won\'t work|stopped working)\b')
        type_e.add_pattern(r'\b(stack trace|traceback|error message|exception thrown)\b')
        type_e.add_pattern(r'\b(resolve|diagnose|investigate) (?:error|issue|problem)\b')
        type_e.add_pattern(r'\bcrashes?\b')
        type_e.add_pattern(r'\b(application|app|program|script) (?:crashes?|fails?|errors?)\b')
        type_e.add_pattern(r'\b(?:debug|fix|resolve).+error\b')
        self.pattern_sets["TYPE_E"] = type_e
        
        # Type D - Web/Testing
        type_d = PatternSet("TYPE_D", description="Web/Testing patterns")
        type_d.add_pattern(r'\b(test|testing|validate|validation|verify|UI|browser|selenium|playwright)\b')
        type_d.add_pattern(r'\b(user journey|e2e|end-to-end|integration test|acceptance test)\b')
        type_d.add_pattern(r'\b(click|navigate|form|button|element|page|website|web app)\b')
        type_d.add_pattern(r'\b(automation|automate|simulate|interaction)\b')
        self.pattern_sets["TYPE_D"] = type_d
        
        # Type C - Research Required
        type_c = PatternSet("TYPE_C", description="Research Required patterns")
        type_c.add_pattern(r'\b(research|search|find|lookup|documentation|docs|API reference)\b')
        type_c.add_pattern(r'\b(current|latest|recent|today|trending|best practices?|state.of.the.art)\b')
        type_c.add_pattern(r'\b(library|framework|package|module|dependency) (?:for|to|that)\b')
        type_c.add_pattern(r'\b(compare|versus|vs|alternatives?|options?|choices?)\b')
        type_c.add_pattern(r'\b(look up|look for|find out) (?:how|what|where|when|why)\b')
        type_c.add_pattern(r'\b(?:how to|guide|tutorial|example) (?:implement|use|configure)\b')
        type_c.add_pattern(r'^what causes\b')
        type_c.add_pattern(r'\b(causes?|reasons?|sources?) (?:of|for)\b')
        self.pattern_sets["TYPE_C"] = type_c
        
        # Type B - Complex/Multi-step
        type_b = PatternSet("TYPE_B", description="Complex/Multi-step patterns")
        type_b.add_pattern(r'\b(implement|build|create|develop|design|architect|refactor)\b')
        type_b.add_pattern(r'\b(feature|functionality|system|module|component|service)\b')
        type_b.add_pattern(r'\b(integrate|migration|upgrade|optimize|scale)\b')
        type_b.add_pattern(r'\b(step.by.step|workflow|process|pipeline|orchestrat)\b')
        type_b.add_pattern(r'\b(multiple|several|various|complex|comprehensive)\b')
        self.pattern_sets["TYPE_B"] = type_b
        
        # Type A - Simple/Direct
        type_a = PatternSet("TYPE_A", description="Simple/Direct patterns")
        type_a.add_pattern(r'^what is (?:a |an |the )?\w+\??$')
        type_a.add_pattern(r'^(when|where|who|why) (?:is|are|was|were|does|do)\s+\w+\??$')
        type_a.add_pattern(r'^what (?:is|are) (?:the )?(?:REST|API|HTTP|HTTPS|JSON|XML|SQL|HTML|CSS)\??$')
        type_a.add_pattern(r'^how does \w+ work\??$')
        type_a.add_pattern(r'^is \w+ \w+\??$')
        type_a.add_pattern(r'^(?:define|explain) \w+$')
        type_a.add_pattern(r'^what time')
        self.pattern_sets["TYPE_A"] = type_a
    
    def _initialize_agent_patterns(self) -> None:
        """Initialize patterns for agent-specific matching"""
        
        # PE Agent patterns
        pe_patterns = PatternSet("PE", description="Prompt Enhancement patterns")
        pe_patterns.add_pattern(r'\b(enhance|improve|clarify|rewrite|rephrase)\b')
        pe_patterns.add_pattern(r'\b(unclear|ambiguous|vague|confusing)\b')
        pe_patterns.add_pattern(r'\b(what do you mean|can you explain|help me understand)\b')
        self.pattern_sets["AGENT_PE"] = pe_patterns
        
        # R1 Agent patterns
        r1_patterns = PatternSet("R1", description="Research Agent patterns")
        r1_patterns.add_pattern(r'\b(search|find|lookup|research|investigate)\b')
        r1_patterns.add_pattern(r'\b(information|data|facts|details|documentation)\b')
        r1_patterns.add_pattern(r'\b(source|reference|citation|link|resource)\b')
        self.pattern_sets["AGENT_R1"] = r1_patterns
        
        # A1 Agent patterns
        a1_patterns = PatternSet("A1", description="Analysis Agent patterns")
        a1_patterns.add_pattern(r'\b(analyze|reason|think|solve|deduce)\b')
        a1_patterns.add_pattern(r'\b(logic|reasoning|analysis|solution|approach)\b')
        a1_patterns.add_pattern(r'\b(why|how|explain the reasoning|walk me through)\b')
        self.pattern_sets["AGENT_A1"] = a1_patterns
        
        # T1 Agent patterns
        t1_patterns = PatternSet("T1", description="Tool Agent patterns")
        t1_patterns.add_pattern(r'\b(tool|utility|script|automation|command)\b')
        t1_patterns.add_pattern(r'\b(execute|run|perform|automate|operate)\b')
        t1_patterns.add_pattern(r'\b(browser|selenium|playwright|api|integration)\b')
        self.pattern_sets["AGENT_T1"] = t1_patterns
        
        # E1 Agent patterns
        e1_patterns = PatternSet("E1", description="Evaluation Agent patterns")
        e1_patterns.add_pattern(r'\b(evaluate|assess|review|check|validate)\b')
        e1_patterns.add_pattern(r'\b(quality|accuracy|correctness|validity|reliability)\b')
        e1_patterns.add_pattern(r'\b(score|rating|assessment|evaluation|judgment)\b')
        self.pattern_sets["AGENT_E1"] = e1_patterns
    
    def get_pattern_set(self, name: str) -> PatternSet | None:
        """Get a specific pattern set by name"""
        return self.pattern_sets.get(name)
    
    def get_classification_patterns(self, task_type: str) -> List[re.Pattern]:
        """Get patterns for a specific classification type"""
        pattern_set = self.pattern_sets.get(task_type)
        return pattern_set.patterns if pattern_set else []
    
    def get_agent_patterns(self, agent_name: str) -> List[re.Pattern]:
        """Get patterns for a specific agent"""
        pattern_set = self.pattern_sets.get(f"AGENT_{agent_name}")
        return pattern_set.patterns if pattern_set else []
    
    def match_patterns(self, text: str, pattern_set_name: str) -> List[str]:
        """Match text against a specific pattern set"""
        pattern_set = self.pattern_sets.get(pattern_set_name)
        if pattern_set:
            return pattern_set.match(text)
        return []
    
    def get_all_pattern_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about all pattern sets"""
        stats = {}
        for name, pattern_set in self.pattern_sets.items():
            stats[name] = {
                "pattern_count": len(pattern_set.patterns),
                "description": pattern_set.description,
                "task_type": pattern_set.task_type
            }
        return stats