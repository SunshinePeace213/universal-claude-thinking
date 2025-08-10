"""
Performance Monitoring Dashboard Queries
SQL queries and views for delegation metrics visualization
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DashboardQueries:
    """
    Collection of SQL queries for performance monitoring dashboard.
    Provides real-time insights into classification and delegation performance.
    """
    
    @staticmethod
    def get_classification_accuracy_query() -> str:
        """Query for classification accuracy over time"""
        return """
        SELECT 
            DATE(created_at) as date,
            predicted_type as task_type,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_percentage,
            ROUND(AVG(confidence), 3) as avg_confidence,
            ROUND(AVG(processing_time_ms), 2) as avg_processing_time_ms
        FROM classification_history
        WHERE created_at >= datetime('now', '-30 days')
        GROUP BY DATE(created_at), predicted_type
        ORDER BY date DESC, task_type;
        """
    
    @staticmethod
    def get_delegation_performance_query() -> str:
        """Query for delegation method performance"""
        return """
        SELECT 
            delegation_method,
            COUNT(*) as total_delegations,
            ROUND(AVG(confidence_score), 3) as avg_confidence,
            ROUND(AVG(total_latency_ms), 2) as avg_latency_ms,
            ROUND(MIN(total_latency_ms), 2) as min_latency_ms,
            ROUND(MAX(total_latency_ms), 2) as max_latency_ms,
            ROUND(AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate
        FROM delegation_metrics
        WHERE created_at >= datetime('now', '-7 days')
        GROUP BY delegation_method
        ORDER BY total_delegations DESC;
        """
    
    @staticmethod
    def get_stage_latency_breakdown_query() -> str:
        """Query for stage-wise latency breakdown"""
        return """
        SELECT 
            strftime('%H:00', created_at) as hour,
            ROUND(AVG(stage1_latency_ms), 2) as avg_keyword_latency,
            ROUND(AVG(stage2_latency_ms), 2) as avg_semantic_latency,
            ROUND(AVG(stage3_latency_ms), 2) as avg_fallback_latency,
            ROUND(AVG(total_latency_ms), 2) as avg_total_latency,
            COUNT(*) as request_count
        FROM delegation_metrics
        WHERE created_at >= datetime('now', '-24 hours')
        GROUP BY strftime('%H:00', created_at)
        ORDER BY hour DESC;
        """
    
    @staticmethod
    def get_agent_utilization_query() -> str:
        """Query for agent utilization patterns"""
        return """
        SELECT 
            selected_agent,
            classification_type,
            COUNT(*) as delegation_count,
            ROUND(AVG(confidence_score), 3) as avg_confidence,
            ROUND(AVG(total_latency_ms), 2) as avg_latency_ms,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_completions,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_completions
        FROM delegation_metrics
        WHERE created_at >= datetime('now', '-7 days')
        GROUP BY selected_agent, classification_type
        ORDER BY delegation_count DESC;
        """
    
    @staticmethod
    def get_confidence_distribution_query() -> str:
        """Query for confidence score distribution"""
        return """
        SELECT 
            CASE 
                WHEN confidence_score >= 0.9 THEN 'Very High (≥0.9)'
                WHEN confidence_score >= 0.8 THEN 'High (0.8-0.9)'
                WHEN confidence_score >= 0.7 THEN 'Medium (0.7-0.8)'
                WHEN confidence_score >= 0.6 THEN 'Low (0.6-0.7)'
                ELSE 'Very Low (<0.6)'
            END as confidence_band,
            COUNT(*) as count,
            ROUND(AVG(total_latency_ms), 2) as avg_latency_ms,
            delegation_method
        FROM delegation_metrics
        WHERE created_at >= datetime('now', '-7 days')
        GROUP BY confidence_band, delegation_method
        ORDER BY confidence_band DESC, count DESC;
        """
    
    @staticmethod
    def get_error_analysis_query() -> str:
        """Query for error and failure analysis"""
        return """
        SELECT 
            classification_type,
            delegation_method,
            error_message,
            COUNT(*) as error_count,
            MIN(created_at) as first_occurrence,
            MAX(created_at) as last_occurrence
        FROM delegation_metrics
        WHERE success = 0 
            AND created_at >= datetime('now', '-7 days')
            AND error_message IS NOT NULL
        GROUP BY classification_type, delegation_method, error_message
        ORDER BY error_count DESC;
        """
    
    @staticmethod
    def get_pattern_effectiveness_query() -> str:
        """Query for pattern matching effectiveness"""
        return """
        SELECT 
            predicted_type,
            patterns_matched,
            COUNT(*) as match_count,
            ROUND(AVG(confidence), 3) as avg_confidence,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_percentage
        FROM classification_history
        WHERE created_at >= datetime('now', '-7 days')
            AND patterns_matched IS NOT NULL
        GROUP BY predicted_type, patterns_matched
        ORDER BY match_count DESC
        LIMIT 50;
        """
    
    @staticmethod
    def get_hourly_throughput_query() -> str:
        """Query for hourly request throughput"""
        return """
        SELECT 
            strftime('%Y-%m-%d %H:00', created_at) as hour,
            COUNT(*) as total_requests,
            ROUND(AVG(total_latency_ms), 2) as avg_latency_ms,
            ROUND(MIN(total_latency_ms), 2) as min_latency_ms,
            ROUND(MAX(total_latency_ms), 2) as max_latency_ms,
            ROUND(COUNT(*) * 1.0 / 3600 * 1000, 2) as requests_per_second
        FROM delegation_metrics
        WHERE created_at >= datetime('now', '-24 hours')
        GROUP BY strftime('%Y-%m-%d %H:00', created_at)
        ORDER BY hour DESC;
        """
    
    @staticmethod
    def get_confidence_factors_analysis_query() -> str:
        """Query for confidence factor analysis"""
        return """
        SELECT 
            ROUND(AVG(classification_score), 3) as avg_classification_score,
            ROUND(AVG(keyword_match_score), 3) as avg_keyword_score,
            ROUND(AVG(semantic_similarity_score), 3) as avg_semantic_score,
            ROUND(AVG(context_quality_score), 3) as avg_context_score,
            ROUND(AVG(input_clarity_score), 3) as avg_clarity_score,
            ROUND(AVG(overall_confidence), 3) as avg_overall_confidence,
            COUNT(*) as sample_count
        FROM confidence_factors
        WHERE created_at >= datetime('now', '-7 days');
        """
    
    @staticmethod
    def get_performance_trend_query() -> str:
        """Query for performance trends over time"""
        return """
        WITH daily_stats AS (
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as daily_requests,
                ROUND(AVG(total_latency_ms), 2) as avg_latency,
                ROUND(AVG(confidence_score), 3) as avg_confidence,
                SUM(CASE WHEN total_latency_ms < 10 THEN 1 ELSE 0 END) as under_10ms_count,
                SUM(CASE WHEN total_latency_ms < 100 THEN 1 ELSE 0 END) as under_100ms_count
            FROM delegation_metrics
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY DATE(created_at)
        )
        SELECT 
            date,
            daily_requests,
            avg_latency,
            avg_confidence,
            ROUND(under_10ms_count * 100.0 / daily_requests, 2) as pct_under_10ms,
            ROUND(under_100ms_count * 100.0 / daily_requests, 2) as pct_under_100ms
        FROM daily_stats
        ORDER BY date DESC;
        """
    
    @staticmethod
    def create_dashboard_views(db_connection) -> None:
        """Create database views for dashboard queries"""
        views = [
            (
                "v_classification_accuracy",
                DashboardQueries.get_classification_accuracy_query()
            ),
            (
                "v_delegation_performance", 
                DashboardQueries.get_delegation_performance_query()
            ),
            (
                "v_stage_latency",
                DashboardQueries.get_stage_latency_breakdown_query()
            ),
            (
                "v_agent_utilization",
                DashboardQueries.get_agent_utilization_query()
            ),
            (
                "v_confidence_distribution",
                DashboardQueries.get_confidence_distribution_query()
            ),
            (
                "v_hourly_throughput",
                DashboardQueries.get_hourly_throughput_query()
            ),
            (
                "v_performance_trends",
                DashboardQueries.get_performance_trend_query()
            )
        ]
        
        cursor = db_connection.cursor()
        
        for view_name, query in views:
            try:
                # Drop view if exists
                cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
                
                # Create view
                create_view_sql = f"CREATE VIEW {view_name} AS {query}"
                cursor.execute(create_view_sql)
                
                logger.info(f"Created view: {view_name}")
                
            except Exception as e:
                logger.error(f"Error creating view {view_name}: {e}")
        
        db_connection.commit()
    
    @staticmethod
    def get_dashboard_summary() -> Dict[str, str]:
        """Get a summary of all available dashboard queries"""
        return {
            "classification_accuracy": "Track classification accuracy by task type over time",
            "delegation_performance": "Analyze performance of different delegation methods",
            "stage_latency": "Breakdown of latency by delegation stage",
            "agent_utilization": "Agent usage patterns and success rates",
            "confidence_distribution": "Distribution of confidence scores across delegations",
            "error_analysis": "Analysis of errors and failures",
            "pattern_effectiveness": "Effectiveness of pattern matching",
            "hourly_throughput": "Request throughput by hour",
            "confidence_factors": "Analysis of confidence scoring factors",
            "performance_trends": "Long-term performance trends"
        }