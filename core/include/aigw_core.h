/*
 * C interface of libaigw_core.
 *
 * The Lua binding in gateway/lua/aigw/core.lua repeats these declarations in
 * ffi.cdef(). If you change this file, change that binding in the same commit.
 * The struct layout is checked on both sides: core/src/lib.rs (Rust test) and
 * tests/unit/core_spec.lua (LuaJIT test).
 *
 * Contract for every function:
 *   - It runs on an NGINX worker event loop, so it must return in microseconds.
 *   - It never unwinds a Rust panic into the caller. A panic becomes an error code.
 *   - Text is passed as a pointer and a byte length. It does not need a NUL byte.
 *     Invalid UTF-8 is replaced, not rejected.
 */
#ifndef AIGW_CORE_H
#define AIGW_CORE_H

#include <stddef.h>
#include <stdint.h>

#define AIGW_OK 0
#define AIGW_ERR_NULL (-1)
#define AIGW_ERR_PANIC (-2)

#define AIGW_TIER_CHEAP 0u
#define AIGW_TIER_STRONG 1u

typedef struct {
    uint32_t tier;          /* AIGW_TIER_CHEAP or AIGW_TIER_STRONG */
    float score;            /* 0.0 (easy) to 1.0 (hard) */
    uint32_t prompt_tokens; /* estimated tokens of the routed text */
} aigw_route_result;

/* Library version as a static NUL-terminated string. Do not free it. */
const char *aigw_version(void);

/* Heuristic token estimate. Returns 0 for a NULL pointer. */
uint32_t aigw_estimate_tokens(const char *text, size_t len);

/*
 * Pick a model tier for a prompt. The result goes to *out.
 * threshold: score at or above which the strong tier is chosen.
 * A threshold that is not finite falls back to 0.5.
 * Returns AIGW_OK, AIGW_ERR_NULL or AIGW_ERR_PANIC.
 */
int32_t aigw_route(const char *text, size_t len, float threshold, aigw_route_result *out);

/*
 * Scan a prompt with the regex deny list of the guardrail. Writes the index
 * of the first matching rule to *rule, or -1 when no rule matches.
 * Returns AIGW_OK, AIGW_ERR_NULL or AIGW_ERR_PANIC.
 */
int32_t aigw_guard(const char *text, size_t len, int32_t *rule);

/* Name of a guard rule as a static NUL-terminated string, or NULL. Do not free it. */
const char *aigw_guard_rule_name(int32_t rule);

#endif /* AIGW_CORE_H */
