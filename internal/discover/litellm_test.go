package discover

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/catwalk/pkg/catwalk"
	"github.com/stretchr/testify/require"
)

func TestLitellmEnricher(t *testing.T) {
	t.Parallel()

	t.Run("populates metadata from /model/info", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/model/info", r.URL.Path)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"data": [
					{
						"model_name": "gpt-4o",
						"model_info": {
							"max_input_tokens": 128000,
							"max_output_tokens": 16384,
							"input_cost_per_token": 2.5e-06,
							"output_cost_per_token": 1e-05
						}
					},
					{
						"model_name": "claude-3-opus",
						"model_info": {
							"max_input_tokens": 200000,
							"max_output_tokens": 4096
						}
					}
				]
			}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID: "test-litellm",
			// Base URL includes /v1 (as Crush configures it); the
			// enricher strips it so /model/info resolves at the root.
			BaseURL: srv.URL + "/v1",
			APIKey:  "test-key",
		}
		models := []catwalk.Model{
			{ID: "gpt-4o", Name: "gpt-4o"},
			{ID: "claude-3-opus", Name: "claude-3-opus"},
			{ID: "unknown-model", Name: "unknown-model"},
		}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)
		require.Len(t, result, 3)

		// gpt-4o should have all fields populated.
		require.Equal(t, int64(128000), result[0].ContextWindow)
		require.Equal(t, int64(16384), result[0].DefaultMaxTokens)
		require.InDelta(t, 2.5, result[0].CostPer1MIn, 0.01)
		require.InDelta(t, 10.0, result[0].CostPer1MOut, 0.01)

		// claude-3-opus should have context window and max tokens.
		require.Equal(t, int64(200000), result[1].ContextWindow)
		require.Equal(t, int64(4096), result[1].DefaultMaxTokens)

		// unknown-model should be unchanged.
		require.Equal(t, int64(0), result[2].ContextWindow)
	})

	t.Run("preserves existing non-zero values", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"data": [{
					"model_name": "gpt-4o",
					"model_info": {
						"max_input_tokens": 128000,
						"max_output_tokens": 16384
					}
				}]
			}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
		}
		models := []catwalk.Model{
			{ID: "gpt-4o", Name: "GPT-4o Custom", ContextWindow: 200000, DefaultMaxTokens: 32768},
		}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)

		// User overrides should be preserved.
		require.Equal(t, int64(200000), result[0].ContextWindow)
		require.Equal(t, int64(32768), result[0].DefaultMaxTokens)
		require.Equal(t, "GPT-4o Custom", result[0].Name)
	})

	t.Run("populates vision and reasoning capabilities", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"data": [
					{
						"model_name": "gpt-4o",
						"model_info": {
							"supports_vision": true,
							"supports_reasoning": true
						}
					},
					{
						"model_name": "text-only",
						"model_info": {}
					}
				]
			}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
		}
		models := []catwalk.Model{
			{ID: "gpt-4o", Name: "gpt-4o"},
			{ID: "text-only", Name: "text-only"},
		}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)

		require.True(t, result[0].SupportsImages)
		require.True(t, result[0].CanReason)

		require.False(t, result[1].SupportsImages)
		require.False(t, result[1].CanReason)
	})

	t.Run("does not override user-set capabilities", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"data": [{
					"model_name": "gpt-4o",
					"model_info": {
						"supports_vision": false,
						"supports_reasoning": false
					}
				}]
			}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
		}
		models := []catwalk.Model{
			{ID: "gpt-4o", Name: "gpt-4o", SupportsImages: true, CanReason: true},
		}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)

		require.True(t, result[0].SupportsImages)
		require.True(t, result[0].CanReason)
	})

	t.Run("populates cached pricing", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"data": [{
					"model_name": "claude-sonnet",
					"model_info": {
						"input_cost_per_token": 3e-06,
						"output_cost_per_token": 1.5e-05,
						"cache_creation_input_token_cost": 3.75e-06,
						"cache_read_input_token_cost": 3e-07
					}
				}]
			}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
		}
		models := []catwalk.Model{{ID: "claude-sonnet", Name: "claude-sonnet"}}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)

		require.InDelta(t, 3.75, result[0].CostPer1MInCached, 0.01)
		require.InDelta(t, 0.3, result[0].CostPer1MOutCached, 0.01)
	})

	t.Run("populates reasoning levels and default effort", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"data": [{
					"model_name": "glm-5.3",
					"model_info": {
						"supports_reasoning": true,
						"reasoning_effort_levels": ["low", "high", "max"],
						"default_reasoning_effort": "high"
					}
				}]
			}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
		}
		models := []catwalk.Model{{ID: "glm-5.3", Name: "glm-5.3"}}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)

		require.True(t, result[0].CanReason)
		require.Equal(t, []string{"low", "high", "max"}, result[0].ReasoningLevels)
		require.Equal(t, "high", result[0].DefaultReasoningEffort)
	})

	t.Run("falls back to max_tokens for max output tokens", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"data": [
					{
						"model_name": "legacy-model",
						"model_info": {"max_tokens": 8192}
					},
					{
						"model_name": "modern-model",
						"model_info": {"max_tokens": 8192, "max_output_tokens": 4096}
					}
				]
			}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
		}
		models := []catwalk.Model{
			{ID: "legacy-model", Name: "legacy-model"},
			{ID: "modern-model", Name: "modern-model"},
		}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)

		require.Equal(t, int64(8192), result[0].DefaultMaxTokens)
		require.Equal(t, int64(4096), result[1].DefaultMaxTokens)
	})

	t.Run("returns models unchanged on HTTP error", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
		}
		models := []catwalk.Model{{ID: "m1"}}

		e := &litellmEnricher{}
		result, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, models)
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.Equal(t, "m1", result[0].ID)
	})

	t.Run("sends auth and extra headers", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "Bearer my-key", r.Header.Get("Authorization"))
			require.Equal(t, "custom-val", r.Header.Get("X-Custom"))
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data": []}`))
		}))
		defer srv.Close()

		cfg := Config{
			ID:      "test-litellm",
			BaseURL: srv.URL,
			APIKey:  "my-key",
			ExtraHeaders: map[string]string{
				"X-Custom": "custom-val",
			},
		}

		e := &litellmEnricher{}
		_, err := e.EnrichModels(context.Background(), cfg, &mockResolver{}, nil)
		require.NoError(t, err)
	})
}

func TestGetEnricher(t *testing.T) {
	t.Parallel()

	require.NotNil(t, GetEnricher("litellm"))
	require.Nil(t, GetEnricher("openai-compat"))
	require.Nil(t, GetEnricher(""))
}

func TestIsKnownCustomProvider(t *testing.T) {
	t.Parallel()

	require.True(t, IsKnownCustomProvider("litellm"))
	require.True(t, IsKnownCustomProvider("ollama"))
	require.True(t, IsKnownCustomProvider("omlx"))
	require.True(t, IsKnownCustomProvider("lmstudio"))
	require.False(t, IsKnownCustomProvider("openai-compat"))
	require.False(t, IsKnownCustomProvider("anthropic"))
	require.False(t, IsKnownCustomProvider(""))
}
