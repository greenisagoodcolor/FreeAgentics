/**
 * Comprehensive tests for lib modules to increase coverage
 */

import { cn, extractTagsFromMarkdown, formatTimestamp } from '@/lib/utils';
import { encrypt, decrypt } from '@/lib/encryption';
import { LLMClient } from '@/lib/llm-client';
import { LLMError, RateLimitError, AuthenticationError } from '@/lib/llm-errors';
import { validateInput, sanitizeOutput, checkPermissions } from '@/lib/security';
import { DataValidationStorage } from '@/lib/storage/data-validation-storage';
import * as api from '@/lib/api';

// Mock fetch
global.fetch = jest.fn();

describe('Lib Modules Comprehensive Coverage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Utils', () => {
    test('cn combines classnames correctly', () => {
      expect(cn('a', 'b')).toBe('a b');
      expect(cn('a', null, 'b')).toBe('a b');
      expect(cn('a', undefined, 'b')).toBe('a b');
      expect(cn()).toBe('');
    });

    test('extractTagsFromMarkdown works', () => {
      const markdown = '# Title\n\nContent with #tag1 and #tag2';
      const tags = extractTagsFromMarkdown(markdown);
      expect(tags).toContain('tag1');
      expect(tags).toContain('tag2');
    });

    test('formatTimestamp formats dates', () => {
      const date = new Date('2024-01-01T12:00:00Z');
      const formatted = formatTimestamp(date);
      expect(formatted).toContain('2024');
    });
  });

  describe('Encryption', () => {
    test('encrypt and decrypt work together', async () => {
      const text = 'secret message';
      const encrypted = await encrypt(text);
      expect(encrypted).not.toBe(text);
      
      const decrypted = await decrypt(encrypted);
      expect(decrypted).toBe(text);
    });

    test('handles empty strings', async () => {
      const encrypted = await encrypt('');
      expect(encrypted).toBe('');
      
      const decrypted = await decrypt('');
      expect(decrypted).toBe('');
    });
  });

  describe('LLM Client', () => {
    test('creates client with config', () => {
      const client = new LLMClient({ provider: 'openai', apiKey: 'test-key' });
      expect(client.provider).toBe('openai');
    });

    test('sends chat messages', async () => {
      const client = new LLMClient({ provider: 'openai', apiKey: 'test-key' });
      
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ 
          choices: [{ message: { content: 'Response' } }] 
        })
      });

      const response = await client.chat([
        { role: 'user', content: 'Hello' }
      ]);
      
      expect(response).toBe('Response');
    });

    test('handles errors correctly', async () => {
      const client = new LLMClient({ provider: 'openai', apiKey: 'test-key' });
      
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 429,
        statusText: 'Rate Limited'
      });

      await expect(client.chat([{ role: 'user', content: 'Hello' }]))
        .rejects.toThrow(RateLimitError);
    });
  });

  describe('Security', () => {
    test('validates input', () => {
      expect(validateInput('safe input')).toBe(true);
      expect(validateInput('<script>alert("xss")</script>')).toBe(false);
      expect(validateInput('')).toBe(false);
    });

    test('sanitizes output', () => {
      expect(sanitizeOutput('normal text')).toBe('normal text');
      expect(sanitizeOutput('<script>bad</script>')).toBe('');
      expect(sanitizeOutput('text with <b>html</b>')).toBe('text with html');
    });

    test('checks permissions', () => {
      expect(checkPermissions('read', { role: 'admin' })).toBe(true);
      expect(checkPermissions('write', { role: 'viewer' })).toBe(false);
      expect(checkPermissions('delete', { role: 'editor' })).toBe(false);
    });
  });

  describe('Data Validation Storage', () => {
    test('creates storage instance', () => {
      const storage = new DataValidationStorage('test-db');
      expect(storage).toBeDefined();
      expect(storage.dbName).toBe('test-db');
    });

    test('validates data before storage', () => {
      const storage = new DataValidationStorage('test-db');
      
      expect(storage.isValid({ id: 1, name: 'test' })).toBe(true);
      expect(storage.isValid(null)).toBe(false);
      expect(storage.isValid(undefined)).toBe(false);
      expect(storage.isValid('')).toBe(false);
    });

    test('stores and retrieves data', async () => {
      const storage = new DataValidationStorage('test-db');
      const data = { id: 1, name: 'test item' };
      
      await storage.store('items', data);
      const retrieved = await storage.get('items', 1);
      
      expect(retrieved).toEqual(data);
    });
  });

  describe('API Client', () => {
    test('makes API calls', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, data: 'result' })
      });

      const result = await api.apiClient.get('/test');
      expect(result.data).toBe('result');
    });

    test('handles API errors', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });

      await expect(api.apiClient.get('/test'))
        .rejects.toThrow('API Error: 500');
    });

    test('includes auth headers', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      await api.apiClient.post('/test', { data: 'test' }, {
        headers: { 'Authorization': 'Bearer token' }
      });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer token'
          })
        })
      );
    });
  });

  describe('Error Handling', () => {
    test('LLMError has correct properties', () => {
      const error = new LLMError('Test error', 'TEST_CODE');
      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_CODE');
      expect(error.name).toBe('LLMError');
    });

    test('RateLimitError extends LLMError', () => {
      const error = new RateLimitError('Too many requests');
      expect(error).toBeInstanceOf(LLMError);
      expect(error.code).toBe('RATE_LIMIT');
    });

    test('AuthenticationError extends LLMError', () => {
      const error = new AuthenticationError('Invalid API key');
      expect(error).toBeInstanceOf(LLMError);
      expect(error.code).toBe('AUTH_ERROR');
    });
  });
});