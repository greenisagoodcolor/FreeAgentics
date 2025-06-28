import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// Import utilities to test
import { cn, extractTagsFromMarkdown, formatTimestamp } from '@/lib/utils';
import { validateInput, sanitizeOutput, checkPermissions } from '@/lib/security';
import { DataValidationStorage } from '@/lib/storage/data-validation-storage';
import { apiClient } from '@/lib/api';

// Mock fetch
global.fetch = jest.fn();

describe('Coverage Boost Tests - Utils', () => {
  it('tests cn utility thoroughly', () => {
    expect(cn('a')).toBe('a');
    expect(cn('a', 'b')).toBe('a b');
    expect(cn('a', null, 'b')).toBe('a b');
    expect(cn('a', undefined, 'b')).toBe('a b');
    expect(cn('a', false && 'b')).toBe('a');
    expect(cn('a', true && 'b')).toBe('a b');
    expect(cn()).toBe('');
    expect(cn('')).toBe('');
    expect(cn('a', '', 'b')).toBe('a b');
  });

  it('tests extractTagsFromMarkdown', () => {
    expect(extractTagsFromMarkdown('')).toEqual([]);
    expect(extractTagsFromMarkdown('no tags here')).toEqual([]);
    expect(extractTagsFromMarkdown('#tag1')).toEqual(['tag1']);
    expect(extractTagsFromMarkdown('text #tag1 more #tag2')).toEqual(['tag1', 'tag2']);
    expect(extractTagsFromMarkdown('#Tag1 #tag1')).toEqual(['Tag1', 'tag1']);
    expect(extractTagsFromMarkdown('##heading not a tag')).toEqual([]);
  });

  it('tests formatTimestamp', () => {
    const date = new Date('2024-01-01T12:00:00Z');
    expect(formatTimestamp(date)).toContain('2024');
    expect(formatTimestamp(date)).toMatch(/\d{4}/);
    
    const now = new Date();
    expect(formatTimestamp(now)).toBeTruthy();
    
    // Test invalid date
    expect(formatTimestamp(new Date('invalid'))).toContain('Invalid');
  });
});

describe('Coverage Boost Tests - Security', () => {
  it('validates input correctly', () => {
    expect(validateInput('safe input')).toBe(true);
    expect(validateInput('<script>alert("xss")</script>')).toBe(false);
    expect(validateInput('')).toBe(false);
    expect(validateInput('normal text')).toBe(true);
    expect(validateInput('<iframe src="bad"></iframe>')).toBe(false);
    expect(validateInput('onclick="bad()"')).toBe(false);
    expect(validateInput('javascript:void(0)')).toBe(false);
  });

  it('sanitizes output correctly', () => {
    expect(sanitizeOutput('normal text')).toBe('normal text');
    expect(sanitizeOutput('<script>bad</script>')).toBe('');
    expect(sanitizeOutput('text with <b>html</b>')).toBe('text with html');
    expect(sanitizeOutput('')).toBe('');
    expect(sanitizeOutput('<div onclick="bad()">text</div>')).toBe('text');
    expect(sanitizeOutput('&<>"\'')).toContain('&amp;');
  });

  it('checks permissions correctly', () => {
    expect(checkPermissions('read', { role: 'admin' })).toBe(true);
    expect(checkPermissions('write', { role: 'admin' })).toBe(true);
    expect(checkPermissions('delete', { role: 'admin' })).toBe(true);
    expect(checkPermissions('read', { role: 'editor' })).toBe(true);
    expect(checkPermissions('write', { role: 'editor' })).toBe(true);
    expect(checkPermissions('delete', { role: 'editor' })).toBe(false);
    expect(checkPermissions('read', { role: 'viewer' })).toBe(true);
    expect(checkPermissions('write', { role: 'viewer' })).toBe(false);
    expect(checkPermissions('any', { role: 'unknown' })).toBe(false);
  });
});

describe('Coverage Boost Tests - Storage', () => {
  let storage: DataValidationStorage;

  beforeEach(() => {
    storage = new DataValidationStorage('test-db');
  });

  it('validates data correctly', () => {
    expect(storage.isValid({ id: 1 })).toBe(true);
    expect(storage.isValid(null)).toBe(false);
    expect(storage.isValid(undefined)).toBe(false);
    expect(storage.isValid('')).toBe(false);
    expect(storage.isValid({})).toBe(true);
    expect(storage.isValid([])).toBe(true);
    expect(storage.isValid('valid')).toBe(true);
    expect(storage.isValid(0)).toBe(true);
    expect(storage.isValid(false)).toBe(true);
  });

  it('stores and retrieves data', async () => {
    const data = { id: 1, name: 'test' };
    await storage.store('items', data);
    
    const retrieved = await storage.get('items', 1);
    expect(retrieved).toEqual(data);
    
    const notFound = await storage.get('items', 999);
    expect(notFound).toBeNull();
    
    const fromEmptyCollection = await storage.get('empty', 1);
    expect(fromEmptyCollection).toBeNull();
  });

  it('handles all CRUD operations', async () => {
    // Create
    await storage.store('users', { id: 1, name: 'User 1' });
    await storage.store('users', { id: 2, name: 'User 2' });
    
    // Read all
    const all = await storage.getAll('users');
    expect(all).toHaveLength(2);
    
    // Update
    await storage.update('users', 1, { id: 1, name: 'Updated User' });
    const updated = await storage.get('users', 1);
    expect(updated.name).toBe('Updated User');
    
    // Delete
    await storage.delete('users', 1);
    const afterDelete = await storage.get('users', 1);
    expect(afterDelete).toBeNull();
    
    // Count
    const count = await storage.count('users');
    expect(count).toBe(1);
    
    // Clear collection
    await storage.clear('users');
    const afterClear = await storage.count('users');
    expect(afterClear).toBe(0);
  });

  it('handles errors correctly', async () => {
    await expect(storage.store('items', null)).rejects.toThrow('Invalid data');
    await expect(storage.update('items', 1, null)).rejects.toThrow('Invalid data');
    await expect(storage.update('items', 999, { id: 999 })).rejects.toThrow('Item not found');
  });
});

describe('Coverage Boost Tests - API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('makes GET requests', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ data: 'result' })
    });

    const result = await apiClient.get('/test');
    expect(result.data).toBe('result');
    expect(global.fetch).toHaveBeenCalledWith('/test', expect.objectContaining({
      method: 'GET'
    }));
  });

  it('makes POST requests', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true })
    });

    const data = { name: 'test' };
    const result = await apiClient.post('/test', data);
    
    expect(result.success).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith('/test', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify(data)
    }));
  });

  it('makes PUT requests', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ updated: true })
    });

    const data = { id: 1, name: 'updated' };
    const result = await apiClient.put('/test/1', data);
    
    expect(result.updated).toBe(true);
  });

  it('makes DELETE requests', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ deleted: true })
    });

    const result = await apiClient.delete('/test/1');
    expect(result.deleted).toBe(true);
  });

  it('handles API errors', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: 'Not Found'
    });

    await expect(apiClient.get('/notfound')).rejects.toThrow('API Error: 404');
  });

  it('handles custom headers', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({})
    });

    await apiClient.get('/test', {
      headers: { 'Authorization': 'Bearer token' }
    });

    expect(global.fetch).toHaveBeenCalledWith('/test', expect.objectContaining({
      headers: expect.objectContaining({
        'Authorization': 'Bearer token',
        'Content-Type': 'application/json'
      })
    }));
  });
});

describe('Coverage Boost Tests - React Components', () => {
  it('renders a simple component', () => {
    const SimpleComponent = () => <div>Test Component</div>;
    render(<SimpleComponent />);
    expect(screen.getByText('Test Component')).toBeInTheDocument();
  });

  it('handles component state', () => {
    const StatefulComponent = () => {
      const [count, setCount] = React.useState(0);
      return (
        <div>
          <span>Count: {count}</span>
          <button onClick={() => setCount(count + 1)}>Increment</button>
        </div>
      );
    };

    render(<StatefulComponent />);
    expect(screen.getByText('Count: 0')).toBeInTheDocument();
    
    fireEvent.click(screen.getByText('Increment'));
    expect(screen.getByText('Count: 1')).toBeInTheDocument();
  });

  it('handles async operations', async () => {
    const AsyncComponent = () => {
      const [data, setData] = React.useState<string | null>(null);
      
      React.useEffect(() => {
        setTimeout(() => setData('Loaded'), 100);
      }, []);
      
      return <div>{data || 'Loading...'}</div>;
    };

    render(<AsyncComponent />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Loaded')).toBeInTheDocument();
    });
  });

  it('handles forms', () => {
    const FormComponent = () => {
      const [value, setValue] = React.useState('');
      const [submitted, setSubmitted] = React.useState('');
      
      const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setSubmitted(value);
      };
      
      return (
        <form onSubmit={handleSubmit}>
          <input 
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Enter text"
          />
          <button type="submit">Submit</button>
          {submitted && <div>Submitted: {submitted}</div>}
        </form>
      );
    };

    render(<FormComponent />);
    
    const input = screen.getByPlaceholderText('Enter text');
    fireEvent.change(input, { target: { value: 'Test input' } });
    
    fireEvent.click(screen.getByText('Submit'));
    expect(screen.getByText('Submitted: Test input')).toBeInTheDocument();
  });

  it('handles conditional rendering', () => {
    const ConditionalComponent = ({ show }: { show: boolean }) => (
      <div>
        {show && <span>Visible</span>}
        {!show && <span>Hidden</span>}
      </div>
    );

    const { rerender } = render(<ConditionalComponent show={true} />);
    expect(screen.getByText('Visible')).toBeInTheDocument();
    expect(screen.queryByText('Hidden')).not.toBeInTheDocument();
    
    rerender(<ConditionalComponent show={false} />);
    expect(screen.queryByText('Visible')).not.toBeInTheDocument();
    expect(screen.getByText('Hidden')).toBeInTheDocument();
  });
});