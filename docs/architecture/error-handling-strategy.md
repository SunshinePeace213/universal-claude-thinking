# Error Handling Strategy

## Error Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant Hook as Claude Hook
    participant Cognitive as Cognitive Layer
    participant SubAgent
    participant ErrorHandler
    participant Memory
    participant Logger
    
    User->>Hook: Request
    Hook->>Cognitive: Process
    
    alt Processing Error
        Cognitive->>ErrorHandler: CognitiveError
        ErrorHandler->>Logger: Log error context
        ErrorHandler->>Memory: Store error pattern
        
        alt Recoverable
            ErrorHandler->>Cognitive: Retry with adjustment
            Cognitive->>SubAgent: Modified request
            SubAgent-->>Cognitive: Success
            Cognitive-->>User: Result with warning
        else Non-Recoverable
            ErrorHandler->>User: Structured error response
        end
    else SubAgent Error
        SubAgent->>ErrorHandler: SubAgentError
        ErrorHandler->>ErrorHandler: Check isolation
        
        alt Isolated Failure
            ErrorHandler->>Cognitive: Use fallback
            Cognitive-->>User: Degraded result
        else Cascade Risk
            ErrorHandler->>User: System error
        end
    end
    
    Note over Memory: Learn from errors
```

## Unified Error Response Format

```python
@dataclass
class CognitiveError(Exception):
    """Base error class for cognitive architecture."""
    
    error_code: str
    message: str
    layer: str
    details: Optional[Dict[str, Any]] = None
    recovery_suggestions: Optional[List[str]] = None
    correlation_id: Optional[str] = None
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to standardized error response."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "layer": self.layer,
                "details": self.details or {},
                "recovery_suggestions": self.recovery_suggestions or [],
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": self.correlation_id or str(uuid.uuid4())
            }
        }
