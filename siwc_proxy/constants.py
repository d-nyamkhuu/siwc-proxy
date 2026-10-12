"""Protocol values from the SIWC token-sharing documentation."""

ISSUER = "https://auth.openai.com"
METADATA_URL = ISSUER + "/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + DIRECT_SCOPE
DYNAMIC_CLIENT = "dynamic_agent_client"
AGENT_NAME = "siwc-proxy"
USAGE_URL = "https://chatgpt.com/settings/usage"

# Seconds before expiry at which an access token is treated as expired.
LEEWAY = 60

# Refresh errors that end the session; anything else keeps saved credentials. `invalid_client` is a
# client-configuration problem rather than a spent token, so it keeps them too.
TERMINAL_REFRESH_ERRORS = frozenset({
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
})

# Admission and response-failure codes and the HTTP status a client should see.
CODE_STATUS = {
    "subscription_sharing_usage_limit_exceeded": 429,
    "subscription_sharing_usage_unavailable": 503,
    "subscription_sharing_user_unavailable": 503,
    "subscription_sharing_user_not_eligible": 403,
    "subscription_sharing_unsupported_capability": 400,
    "subscription_sharing_route_not_supported": 403,
    "subscription_sharing_invalid_user": 401,
}
USAGE_LIMIT_CODE = "subscription_sharing_usage_limit_exceeded"
RETRYABLE_CODES = frozenset({"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"})
