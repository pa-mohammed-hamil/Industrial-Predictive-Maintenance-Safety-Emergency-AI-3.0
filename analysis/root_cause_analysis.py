"""
Automatic Root Cause Analysis (RCA) System

AI-powered root cause identification for equipment failures and incidents.
Uses multi-source data correlation, pattern matching, and causal inference.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict


class CauseCategory(str, Enum):
    """Categories of root causes"""
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    OPERATIONAL = "operational"
    ENVIRONMENTAL = "environmental"
    MATERIAL = "material"
    HUMAN_ERROR = "human_error"
    DESIGN_FLAW = "design_flaw"
    MAINTENANCE = "maintenance"
    SOFTWARE = "software"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence in RCA findings"""
    VERY_HIGH = "very_high"  # >90%
    HIGH = "high"  # 75-90%
    MEDIUM = "medium"  # 50-75%
    LOW = "low"  # 25-50%
    VERY_LOW = "very_low"  # <25%


@dataclass
class Evidence:
    """Evidence supporting a root cause hypothesis"""
    type: str  # telemetry, visual, log, historical, expert
    source: str
    description: str
    timestamp: datetime
    relevance_score: float  # 0-1
    data: Optional[Dict[str, Any]] = None


@dataclass
class RootCause:
    """Identified root cause"""
    cause_id: str
    category: CauseCategory
    description: str
    confidence: float  # 0-1
    confidence_level: ConfidenceLevel
    evidence: List[Evidence] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    similar_incidents: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None
    estimated_time_to_failure: Optional[float] = None  # hours before failure


@dataclass
class CausalChain:
    """Chain of events leading to failure"""
    events: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Tuple[datetime, str]] = field(default_factory=list)
    trigger_event: Optional[str] = None
    propagation_path: List[str] = field(default_factory=list)


@dataclass
class RCAReport:
    """Complete Root Cause Analysis Report"""
    analysis_id: str
    asset_id: str
    incident_id: Optional[str]
    failure_type: str
    failure_timestamp: datetime
    analysis_timestamp: datetime
    
    # Primary findings
    primary_root_cause: Optional[RootCause]
    secondary_root_causes: List[RootCause] = field(default_factory=list)
    
    # Analysis details
    causal_chain: Optional[CausalChain] = None
    contributing_factors: List[str] = field(default_factory=list)
    
    # Evidence
    all_evidence: List[Evidence] = field(default_factory=list)
    data_sources_analyzed: List[str] = field(default_factory=list)
    
    # Recommendations
    corrective_actions: List[str] = field(default_factory=list)
    preventive_measures: List[str] = field(default_factory=list)
    
    # Validation
    overall_confidence: float = 0.0
    validation_status: str = "pending"
    
    # Similar cases
    similar_cases_count: int = 0
    pattern_match_score: float = 0.0
    
    # Cost impact
    estimated_cost_impact: Optional[float] = None
    estimated_downtime: Optional[float] = None
    
    # Metadata
    analysis_duration_seconds: float = 0.0
    ai_models_used: List[str] = field(default_factory=list)


class RootCauseAnalyzer:
    """
    Automatic Root Cause Analysis Engine
    
    Analyzes failures using:
    1. Telemetry pattern analysis
    2. Historical failure correlation
    3. Causal inference models
    4. Expert rules and knowledge base
    5. Multi-source data fusion
    """
    
    def __init__(self, db_session=None, cache_service=None, llm_service=None):
        self.db = db_session
        self.cache = cache_service
        self.llm = llm_service
        
        # Analysis parameters
        self.lookback_hours = 72  # Analyze 72 hours before failure
        self.correlation_threshold = 0.7
        self.confidence_threshold = 0.5
        
        # Knowledge base
        self.failure_patterns = self._load_failure_patterns()
        self.causal_rules = self._load_causal_rules()
    
    async def analyze_failure(
        self,
        asset_id: str,
        failure_timestamp: datetime,
        failure_type: str,
        incident_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RCAReport:
        """
        Perform automatic root cause analysis for a failure
        
        Args:
            asset_id: Asset identifier
            failure_timestamp: When failure occurred
            failure_type: Type of failure
            incident_id: Optional incident reference
            context: Additional context information
            
        Returns:
            Complete RCA report
        """
        start_time = datetime.utcnow()
        analysis_id = f"RCA-{asset_id}-{int(failure_timestamp.timestamp())}"
        
        print(f"🔍 Starting Root Cause Analysis: {analysis_id}")
        
        # Step 1: Collect all relevant data
        data = await self._collect_data(asset_id, failure_timestamp)
        
        # Step 2: Analyze telemetry patterns
        telemetry_evidence = await self._analyze_telemetry(
            data.get("telemetry", []),
            failure_timestamp
        )
        
        # Step 3: Analyze historical patterns
        historical_evidence = await self._analyze_historical_patterns(
            asset_id,
            failure_type,
            data.get("history", [])
        )
        
        # Step 4: Analyze maintenance history
        maintenance_evidence = await self._analyze_maintenance_history(
            data.get("maintenance", []),
            failure_timestamp
        )
        
        # Step 5: Analyze environmental factors
        environmental_evidence = await self._analyze_environmental_factors(
            data.get("environment", []),
            failure_timestamp
        )
        
        # Step 6: Analyze operational context
        operational_evidence = await self._analyze_operational_context(
            data.get("operations", []),
            failure_timestamp
        )
        
        # Step 7: Visual/CV analysis if available
        visual_evidence = await self._analyze_visual_data(
            data.get("images", []),
            failure_type
        )
        
        # Combine all evidence
        all_evidence = (
            telemetry_evidence +
            historical_evidence +
            maintenance_evidence +
            environmental_evidence +
            operational_evidence +
            visual_evidence
        )
        
        # Step 8: Build causal chain
        causal_chain = await self._construct_causal_chain(
            all_evidence,
            failure_timestamp
        )
        
        # Step 9: Identify root causes
        root_causes = await self._identify_root_causes(
            all_evidence,
            causal_chain,
            failure_type
        )
        
        # Step 10: Rank and validate root causes
        ranked_causes = self._rank_root_causes(root_causes, all_evidence)
        
        # Step 11: Find similar cases
        similar_cases = await self._find_similar_cases(
            asset_id,
            failure_type,
            ranked_causes[0] if ranked_causes else None
        )
        
        # Step 12: Generate recommendations
        corrective_actions = self._generate_corrective_actions(ranked_causes)
        preventive_measures = self._generate_preventive_measures(ranked_causes)
        
        # Step 13: Estimate impact
        cost_impact = await self._estimate_cost_impact(
            asset_id,
            failure_type,
            ranked_causes[0] if ranked_causes else None
        )
        
        # Build report
        analysis_duration = (datetime.utcnow() - start_time).total_seconds()
        
        report = RCAReport(
            analysis_id=analysis_id,
            asset_id=asset_id,
            incident_id=incident_id,
            failure_type=failure_type,
            failure_timestamp=failure_timestamp,
            analysis_timestamp=datetime.utcnow(),
            primary_root_cause=ranked_causes[0] if ranked_causes else None,
            secondary_root_causes=ranked_causes[1:5] if len(ranked_causes) > 1 else [],
            causal_chain=causal_chain,
            all_evidence=all_evidence,
            data_sources_analyzed=list(data.keys()),
            corrective_actions=corrective_actions,
            preventive_measures=preventive_measures,
            overall_confidence=ranked_causes[0].confidence if ranked_causes else 0.0,
            similar_cases_count=len(similar_cases),
            estimated_cost_impact=cost_impact.get("total_cost"),
            estimated_downtime=cost_impact.get("downtime_hours"),
            analysis_duration_seconds=analysis_duration,
            ai_models_used=["telemetry_analyzer", "pattern_matcher", "causal_inference"]
        )
        
        # Step 14: Use LLM to enhance report with natural language
        if self.llm:
            report = await self._enhance_with_llm(report)
        
        # Step 15: Save to database
        await self._save_rca_report(report)
        
        print(f"✅ Root Cause Analysis Complete: {analysis_id}")
        print(f"   Primary Cause: {report.primary_root_cause.description if report.primary_root_cause else 'Unknown'}")
        print(f"   Confidence: {report.overall_confidence:.1%}")
        print(f"   Analysis Time: {analysis_duration:.2f}s")
        
        return report
    
    async def _collect_data(
        self,
        asset_id: str,
        failure_timestamp: datetime
    ) -> Dict[str, List[Any]]:
        """Collect all relevant data for analysis"""
        lookback_start = failure_timestamp - timedelta(hours=self.lookback_hours)
        
        data = {
            "telemetry": await self._fetch_telemetry(asset_id, lookback_start, failure_timestamp),
            "history": await self._fetch_failure_history(asset_id),
            "maintenance": await self._fetch_maintenance_records(asset_id, lookback_start),
            "environment": await self._fetch_environmental_data(lookback_start, failure_timestamp),
            "operations": await self._fetch_operational_logs(asset_id, lookback_start),
            "images": await self._fetch_inspection_images(asset_id, lookback_start)
        }
        
        return data
    
    async def _analyze_telemetry(
        self,
        telemetry_data: List[Dict],
        failure_timestamp: datetime
    ) -> List[Evidence]:
        """Analyze sensor telemetry for anomalous patterns"""
        evidence = []
        
        if not telemetry_data:
            return evidence
        
        # Detect anomalies in each parameter
        parameters = self._group_by_parameter(telemetry_data)
        
        for param_name, readings in parameters.items():
            # Check for sudden changes
            sudden_changes = self._detect_sudden_changes(readings, failure_timestamp)
            if sudden_changes:
                evidence.append(Evidence(
                    type="telemetry",
                    source=f"sensor_{param_name}",
                    description=f"Sudden {param_name} change detected: {sudden_changes['description']}",
                    timestamp=sudden_changes["timestamp"],
                    relevance_score=sudden_changes["severity"],
                    data=sudden_changes
                ))
            
            # Check for gradual degradation
            degradation = self._detect_degradation(readings, failure_timestamp)
            if degradation:
                evidence.append(Evidence(
                    type="telemetry",
                    source=f"sensor_{param_name}",
                    description=f"Gradual {param_name} degradation: {degradation['description']}",
                    timestamp=degradation["start_timestamp"],
                    relevance_score=degradation["confidence"],
                    data=degradation
                ))
            
            # Check for threshold violations
            violations = self._detect_threshold_violations(readings, param_name)
            for violation in violations:
                evidence.append(Evidence(
                    type="telemetry",
                    source=f"sensor_{param_name}",
                    description=f"{param_name} exceeded threshold: {violation['description']}",
                    timestamp=violation["timestamp"],
                    relevance_score=0.8,
                    data=violation
                ))
        
        # Multi-parameter correlation
        correlations = self._detect_parameter_correlations(parameters, failure_timestamp)
        for corr in correlations:
            evidence.append(Evidence(
                type="telemetry",
                source="multi_parameter",
                description=f"Correlated anomaly: {corr['description']}",
                timestamp=corr["timestamp"],
                relevance_score=corr["correlation_strength"],
                data=corr
            ))
        
        return evidence
    
    async def _analyze_historical_patterns(
        self,
        asset_id: str,
        failure_type: str,
        history: List[Dict]
    ) -> List[Evidence]:
        """Analyze historical failure patterns"""
        evidence = []
        
        # Find similar past failures
        similar_failures = [
            f for f in history
            if f.get("failure_type") == failure_type
        ]
        
        if similar_failures:
            # Identify common patterns
            common_causes = self._identify_common_causes(similar_failures)
            
            for cause, frequency in common_causes.items():
                confidence = min(frequency / len(similar_failures), 1.0)
                evidence.append(Evidence(
                    type="historical",
                    source="failure_database",
                    description=f"Similar failures ({len(similar_failures)} cases) commonly caused by: {cause}",
                    timestamp=datetime.utcnow(),
                    relevance_score=confidence,
                    data={
                        "cause": cause,
                        "frequency": frequency,
                        "total_cases": len(similar_failures)
                    }
                ))
        
        # Check for recurring issues
        recurring_patterns = self._detect_recurring_patterns(history)
        for pattern in recurring_patterns:
            evidence.append(Evidence(
                type="historical",
                source="pattern_analysis",
                description=f"Recurring pattern detected: {pattern['description']}",
                timestamp=datetime.utcnow(),
                relevance_score=pattern["confidence"],
                data=pattern
            ))
        
        return evidence
    
    async def _analyze_maintenance_history(
        self,
        maintenance_records: List[Dict],
        failure_timestamp: datetime
    ) -> List[Evidence]:
        """Analyze maintenance history for contributing factors"""
        evidence = []
        
        # Check for overdue maintenance
        overdue = self._check_overdue_maintenance(maintenance_records, failure_timestamp)
        if overdue:
            evidence.append(Evidence(
                type="maintenance",
                source="maintenance_schedule",
                description=f"Overdue maintenance detected: {overdue['description']}",
                timestamp=overdue["due_date"],
                relevance_score=0.85,
                data=overdue
            ))
        
        # Check for recent maintenance work
        recent_work = self._check_recent_maintenance(maintenance_records, failure_timestamp)
        if recent_work:
            evidence.append(Evidence(
                type="maintenance",
                source="work_orders",
                description=f"Recent maintenance may be related: {recent_work['description']}",
                timestamp=recent_work["completion_date"],
                relevance_score=0.7,
                data=recent_work
            ))
        
        # Check maintenance quality indicators
        quality_issues = self._analyze_maintenance_quality(maintenance_records)
        for issue in quality_issues:
            evidence.append(Evidence(
                type="maintenance",
                source="quality_analysis",
                description=f"Maintenance quality issue: {issue['description']}",
                timestamp=issue["timestamp"],
                relevance_score=issue["severity"],
                data=issue
            ))
        
        return evidence
    
    async def _analyze_environmental_factors(
        self,
        environmental_data: List[Dict],
        failure_timestamp: datetime
    ) -> List[Evidence]:
        """Analyze environmental conditions"""
        evidence = []
        
        # Check for extreme conditions
        extremes = self._detect_extreme_conditions(environmental_data, failure_timestamp)
        for extreme in extremes:
            evidence.append(Evidence(
                type="environmental",
                source="environmental_monitoring",
                description=f"Extreme condition detected: {extreme['description']}",
                timestamp=extreme["timestamp"],
                relevance_score=extreme["severity"],
                data=extreme
            ))
        
        return evidence
    
    async def _analyze_operational_context(
        self,
        operational_logs: List[Dict],
        failure_timestamp: datetime
    ) -> List[Evidence]:
        """Analyze operational context and operator actions"""
        evidence = []
        
        # Check for abnormal operations
        abnormal_ops = self._detect_abnormal_operations(operational_logs, failure_timestamp)
        for op in abnormal_ops:
            evidence.append(Evidence(
                type="operational",
                source="operation_logs",
                description=f"Abnormal operation detected: {op['description']}",
                timestamp=op["timestamp"],
                relevance_score=op["confidence"],
                data=op
            ))
        
        return evidence
    
    async def _analyze_visual_data(
        self,
        images: List[Dict],
        failure_type: str
    ) -> List[Evidence]:
        """Analyze inspection images and visual data"""
        evidence = []
        
        # Check for visual defects
        for image in images:
            defects = await self._detect_visual_defects(image)
            for defect in defects:
                evidence.append(Evidence(
                    type="visual",
                    source="computer_vision",
                    description=f"Visual defect detected: {defect['type']}",
                    timestamp=image["capture_timestamp"],
                    relevance_score=defect["confidence"],
                    data=defect
                ))
        
        return evidence
    
    async def _construct_causal_chain(
        self,
        evidence: List[Evidence],
        failure_timestamp: datetime
    ) -> CausalChain:
        """Construct timeline of causal events"""
        # Sort evidence by time
        sorted_evidence = sorted(evidence, key=lambda e: e.timestamp)
        
        chain = CausalChain()
        
        # Build timeline
        for e in sorted_evidence:
            chain.timeline.append((e.timestamp, e.description))
            chain.events.append({
                "timestamp": e.timestamp,
                "description": e.description,
                "type": e.type,
                "relevance": e.relevance_score
            })
        
        # Identify trigger event (first significant anomaly)
        trigger = next(
            (e for e in sorted_evidence if e.relevance_score > 0.7),
            None
        )
        if trigger:
            chain.trigger_event = trigger.description
        
        # Build propagation path
        chain.propagation_path = self._infer_propagation_path(sorted_evidence)
        
        return chain
    
    async def _identify_root_causes(
        self,
        evidence: List[Evidence],
        causal_chain: CausalChain,
        failure_type: str
    ) -> List[RootCause]:
        """Identify potential root causes from evidence"""
        root_causes = []
        
        # Group evidence by potential cause
        cause_groups = self._group_evidence_by_cause(evidence)
        
        for cause_key, cause_evidence in cause_groups.items():
            # Calculate confidence based on evidence strength
            confidence = self._calculate_cause_confidence(cause_evidence)
            
            if confidence >= self.confidence_threshold:
                # Determine cause category
                category = self._categorize_cause(cause_evidence, failure_type)
                
                # Generate description
                description = self._generate_cause_description(
                    cause_key,
                    cause_evidence,
                    failure_type
                )
                
                # Extract contributing factors
                contributing_factors = self._extract_contributing_factors(cause_evidence)
                
                root_cause = RootCause(
                    cause_id=f"RC-{cause_key}",
                    category=category,
                    description=description,
                    confidence=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    evidence=cause_evidence,
                    contributing_factors=contributing_factors
                )
                
                root_causes.append(root_cause)
        
        return root_causes
    
    def _rank_root_causes(
        self,
        root_causes: List[RootCause],
        all_evidence: List[Evidence]
    ) -> List[RootCause]:
        """Rank root causes by likelihood and evidence strength"""
        # Score each cause
        for cause in root_causes:
            score = self._score_root_cause(cause, all_evidence)
            cause.confidence = score
            cause.confidence_level = self._get_confidence_level(score)
        
        # Sort by confidence
        return sorted(root_causes, key=lambda c: c.confidence, reverse=True)
    
    async def _find_similar_cases(
        self,
        asset_id: str,
        failure_type: str,
        primary_cause: Optional[RootCause]
    ) -> List[str]:
        """Find similar historical cases"""
        # Search database for similar failures
        similar = []
        
        # TODO: Implement similarity search
        # - Same asset type
        # - Same failure type
        # - Similar root cause
        # - Similar symptom pattern
        
        return similar
    
    def _generate_corrective_actions(
        self,
        root_causes: List[RootCause]
    ) -> List[str]:
        """Generate corrective action recommendations"""
        actions = []
        
        for cause in root_causes[:3]:  # Top 3 causes
            if cause.category == CauseCategory.MECHANICAL:
                actions.append(f"Inspect and repair mechanical component: {cause.description}")
            elif cause.category == CauseCategory.ELECTRICAL:
                actions.append(f"Check electrical connections and components: {cause.description}")
            elif cause.category == CauseCategory.THERMAL:
                actions.append(f"Investigate cooling system and thermal management: {cause.description}")
            elif cause.category == CauseCategory.MAINTENANCE:
                actions.append(f"Update maintenance schedule: {cause.description}")
            elif cause.category == CauseCategory.OPERATIONAL:
                actions.append(f"Review and adjust operational procedures: {cause.description}")
            else:
                actions.append(f"Investigate and address: {cause.description}")
        
        return actions
    
    def _generate_preventive_measures(
        self,
        root_causes: List[RootCause]
    ) -> List[str]:
        """Generate preventive measure recommendations"""
        measures = []
        
        for cause in root_causes[:3]:
            measures.extend([
                f"Implement condition monitoring for: {cause.category.value}",
                f"Establish alerts for early detection of: {cause.description}",
                f"Update preventive maintenance to address: {cause.category.value} issues"
            ])
        
        return measures
    
    async def _estimate_cost_impact(
        self,
        asset_id: str,
        failure_type: str,
        primary_cause: Optional[RootCause]
    ) -> Dict[str, float]:
        """Estimate cost impact of the failure"""
        # TODO: Integrate with cost engine
        return {
            "total_cost": 0.0,
            "downtime_hours": 0.0,
            "repair_cost": 0.0,
            "production_loss": 0.0
        }
    
    async def _enhance_with_llm(self, report: RCAReport) -> RCAReport:
        """Use LLM to enhance report with natural language summaries"""
        if not self.llm or not report.primary_root_cause:
            return report
        
        # Generate enhanced description using LLM
        prompt = self._build_llm_prompt(report)
        enhanced_description = await self.llm.generate(prompt)
        
        if enhanced_description:
            report.primary_root_cause.description = enhanced_description
        
        return report
    
    async def _save_rca_report(self, report: RCAReport):
        """Save RCA report to database"""
        # TODO: Implement database save
        pass
    
    # Helper methods
    
    def _load_failure_patterns(self) -> Dict:
        """Load known failure patterns from knowledge base"""
        return {
            "bearing_failure": {
                "symptoms": ["increased_vibration", "temperature_rise", "noise"],
                "causes": ["lack_of_lubrication", "contamination", "misalignment"]
            },
            "motor_failure": {
                "symptoms": ["overcurrent", "overheating", "vibration"],
                "causes": ["overload", "phase_imbalance", "bearing_failure"]
            }
        }
    
    def _load_causal_rules(self) -> Dict:
        """Load causal inference rules"""
        return {
            "vibration_increase + temperature_increase": "bearing_degradation",
            "current_spike + voltage_drop": "electrical_fault",
            "gradual_performance_degradation": "wear_deterioration"
        }
    
    async def _fetch_telemetry(self, asset_id, start, end):
        """Fetch telemetry data"""
        # TODO: Implement actual database query
        return []
    
    async def _fetch_failure_history(self, asset_id):
        """Fetch failure history"""
        return []
    
    async def _fetch_maintenance_records(self, asset_id, start):
        """Fetch maintenance records"""
        return []
    
    async def _fetch_environmental_data(self, start, end):
        """Fetch environmental data"""
        return []
    
    async def _fetch_operational_logs(self, asset_id, start):
        """Fetch operational logs"""
        return []
    
    async def _fetch_inspection_images(self, asset_id, start):
        """Fetch inspection images"""
        return []
    
    def _group_by_parameter(self, telemetry):
        """Group telemetry by parameter name"""
        grouped = defaultdict(list)
        for reading in telemetry:
            grouped[reading["parameter"]].append(reading)
        return dict(grouped)
    
    def _detect_sudden_changes(self, readings, failure_time):
        """Detect sudden changes in readings"""
        # TODO: Implement change point detection
        return None
    
    def _detect_degradation(self, readings, failure_time):
        """Detect gradual degradation"""
        # TODO: Implement trend analysis
        return None
    
    def _detect_threshold_violations(self, readings, param_name):
        """Detect threshold violations"""
        return []
    
    def _detect_parameter_correlations(self, parameters, failure_time):
        """Detect correlated anomalies across parameters"""
        return []
    
    def _identify_common_causes(self, failures):
        """Identify common causes in failure history"""
        causes = defaultdict(int)
        for failure in failures:
            if "root_cause" in failure:
                causes[failure["root_cause"]] += 1
        return dict(causes)
    
    def _detect_recurring_patterns(self, history):
        """Detect recurring failure patterns"""
        return []
    
    def _check_overdue_maintenance(self, records, failure_time):
        """Check for overdue maintenance"""
        return None
    
    def _check_recent_maintenance(self, records, failure_time):
        """Check recent maintenance work"""
        return None
    
    def _analyze_maintenance_quality(self, records):
        """Analyze maintenance quality"""
        return []
    
    def _detect_extreme_conditions(self, env_data, failure_time):
        """Detect extreme environmental conditions"""
        return []
    
    def _detect_abnormal_operations(self, logs, failure_time):
        """Detect abnormal operations"""
        return []
    
    async def _detect_visual_defects(self, image):
        """Detect defects in images"""
        return []
    
    def _group_evidence_by_cause(self, evidence):
        """Group evidence by potential cause"""
        groups = defaultdict(list)
        for e in evidence:
            # Simple grouping by type and keywords
            key = f"{e.type}_{e.source}"
            groups[key].append(e)
        return dict(groups)
    
    def _calculate_cause_confidence(self, evidence):
        """Calculate confidence for a cause based on evidence"""
        if not evidence:
            return 0.0
        
        # Weighted average of evidence relevance
        total_relevance = sum(e.relevance_score for e in evidence)
        confidence = total_relevance / len(evidence)
        
        # Boost confidence if multiple evidence types
        unique_types = len(set(e.type for e in evidence))
        confidence *= (1 + 0.1 * unique_types)
        
        return min(confidence, 1.0)
    
    def _categorize_cause(self, evidence, failure_type):
        """Categorize the root cause"""
        # Simple heuristic based on evidence types
        types = [e.type for e in evidence]
        
        if "telemetry" in types and any("vibration" in e.description for e in evidence):
            return CauseCategory.MECHANICAL
        elif "telemetry" in types and any("temperature" in e.description for e in evidence):
            return CauseCategory.THERMAL
        elif "telemetry" in types and any("current" in e.description or "voltage" in e.description for e in evidence):
            return CauseCategory.ELECTRICAL
        elif "maintenance" in types:
            return CauseCategory.MAINTENANCE
        elif "operational" in types:
            return CauseCategory.OPERATIONAL
        elif "environmental" in types:
            return CauseCategory.ENVIRONMENTAL
        else:
            return CauseCategory.UNKNOWN
    
    def _generate_cause_description(self, cause_key, evidence, failure_type):
        """Generate human-readable cause description"""
        # Take the most relevant evidence
        top_evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:3]
        descriptions = [e.description for e in top_evidence]
        return "; ".join(descriptions)
    
    def _extract_contributing_factors(self, evidence):
        """Extract contributing factors from evidence"""
        factors = []
        for e in evidence:
            if e.relevance_score > 0.5:
                factors.append(e.description)
        return factors[:5]  # Top 5
    
    def _get_confidence_level(self, confidence):
        """Convert confidence score to level"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _score_root_cause(self, cause, all_evidence):
        """Score a root cause"""
        # Base score is the cause confidence
        score = cause.confidence
        
        # Boost if cause has diverse evidence
        evidence_types = set(e.type for e in cause.evidence)
        score *= (1 + 0.05 * len(evidence_types))
        
        # Boost if cause has high-relevance evidence
        high_relevance = sum(1 for e in cause.evidence if e.relevance_score > 0.8)
        score *= (1 + 0.1 * high_relevance)
        
        return min(score, 1.0)
    
    def _infer_propagation_path(self, evidence):
        """Infer how failure propagated"""
        path = []
        for e in evidence:
            if e.relevance_score > 0.6:
                path.append(f"{e.type}: {e.source}")
        return path
    
    def _build_llm_prompt(self, report):
        """Build prompt for LLM enhancement"""
        return f"""
        Analyze this failure and provide a concise root cause summary:
        
        Asset: {report.asset_id}
        Failure Type: {report.failure_type}
        Evidence:
        {chr(10).join(f"- {e.description}" for e in report.all_evidence[:10])}
        
        Provide a clear, technical explanation of the root cause.
        """


# Export
__all__ = [
    "RootCauseAnalyzer",
    "RCAReport",
    "RootCause",
    "CauseCategory",
    "ConfidenceLevel",
    "Evidence",
    "CausalChain"
]
