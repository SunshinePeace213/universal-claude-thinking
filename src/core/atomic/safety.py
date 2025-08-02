"""Safety validation and input sanitization for atomic prompt analysis."""

import re


class SafetyValidator:
    """Validates prompts for safety and detects potential injection attempts."""

    # Maximum allowed prompt length
    MAX_PROMPT_LENGTH = 10000

    # Patterns that might indicate prompt injection
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        r"ignore\s+(all\s+)?previous\s+(instructions|prompts)",
        r"disregard\s+.*instructions",
        r"forget\s+everything",
        r"new\s+instructions?:\s*",
        r"system\s*:\s*you\s+are",
        # Role playing attempts
        r"you\s+are\s+now\s+[a-z]+",
        r"pretend\s+to\s+be\s+[a-z]+",
        r"act\s+as\s+[a-z]+",
        r"roleplay\s+as",
        # Jailbreak attempts
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode",
        r"unlock\s+.*mode",
        # Command injection
        r"\$\{.*\}",  # Variable expansion
        r"\$\(.*\)",  # Command substitution
        r"`.*`",  # Backtick execution
        r"<script.*>.*</script>",  # Script tags
        # Encoding attempts
        r"base64|atob|btoa",
        r"\\x[0-9a-f]{2}",  # Hex encoding
        r"\\u[0-9a-f]{4}",  # Unicode escape
    ]

    # Suspicious content patterns (not blocking, but increases scrutiny)
    SUSPICIOUS_PATTERNS = [
        r"password|token|api[_\s]?key|secret",
        r"eval\s*\(",
        r"exec\s*\(",
        r"import\s+os",
        r"subprocess",
        r"__.*__",  # Dunder methods
    ]

    def validate_prompt(self, prompt: str) -> tuple[bool, str | None]:
        """Validate prompt for safety concerns.

        Args:
            prompt: Input prompt to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check length
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            return (
                False,
                f"Prompt exceeds maximum length of {self.MAX_PROMPT_LENGTH} characters",
            )

        # Check for empty or whitespace-only
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"

        # Check for injection patterns
        injection_found = self._check_patterns(prompt, self.INJECTION_PATTERNS)
        if injection_found:
            return False, f"Potential prompt injection detected: {injection_found}"

        # Check for suspicious content (warning only)
        suspicious_found = self._check_patterns(prompt, self.SUSPICIOUS_PATTERNS)
        if suspicious_found:
            # Log warning but don't block
            print(f"Warning: Suspicious pattern detected: {suspicious_found}")

        return True, None

    def _check_patterns(self, text: str, patterns: list[str]) -> str | None:
        """Check text against a list of regex patterns.

        Args:
            text: Text to check
            patterns: List of regex patterns

        Returns:
            First matching pattern description, or None
        """
        text_lower = text.lower()

        for pattern in patterns:
            try:
                if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                    # Return a sanitized version of the pattern for error message
                    return pattern.replace("\\", "").replace(".*", "...")[:50]
            except re.error:
                # Skip invalid patterns
                continue

        return None

    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt by removing potentially harmful content.

        Args:
            prompt: Input prompt

        Returns:
            Sanitized prompt
        """
        # Replace null bytes with spaces
        sanitized = prompt.replace("\x00", " ")

        # Remove control characters (except newlines, tabs, and spaces)
        sanitized = "".join(
            char for char in sanitized if char in ["\n", "\t", " "] or ord(char) >= 32
        )

        # Normalize multiple spaces to single space
        sanitized = re.sub(r" +", " ", sanitized)

        # Truncate if too long
        if len(sanitized) > self.MAX_PROMPT_LENGTH:
            sanitized = sanitized[: self.MAX_PROMPT_LENGTH] + "..."

        return sanitized

    def calculate_risk_score(self, prompt: str) -> float:
        """Calculate a risk score for the prompt (0.0 = safe, 1.0 = high risk).

        Args:
            prompt: Input prompt

        Returns:
            Risk score between 0.0 and 1.0
        """
        risk_score = 0.0

        # Check injection patterns (high risk)
        injection_count = sum(
            1
            for pattern in self.INJECTION_PATTERNS
            if re.search(pattern, prompt.lower(), re.IGNORECASE)
        )
        risk_score += min(0.6, injection_count * 0.3)

        # Check suspicious patterns (medium risk)
        suspicious_count = sum(
            1
            for pattern in self.SUSPICIOUS_PATTERNS
            if re.search(pattern, prompt.lower(), re.IGNORECASE)
        )
        risk_score += min(0.4, suspicious_count * 0.15)

        # Length-based risk (very long prompts)
        if len(prompt) > 8000:
            risk_score += 0.2
        elif len(prompt) > 5000:
            risk_score += 0.1

        # Encoding complexity
        special_char_ratio = sum(
            1 for c in prompt if not c.isalnum() and c not in " \n\t.,!?"
        ) / max(len(prompt), 1)
        if special_char_ratio > 0.3:
            risk_score += 0.1

        return min(1.0, risk_score)
