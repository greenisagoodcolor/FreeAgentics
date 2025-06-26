import { test, expect } from '@playwright/test'

test.describe('Performance Tests', () => {
  test('page load performance is acceptable', async ({ page }) => {
    const startTime = Date.now()
    
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    const loadTime = Date.now() - startTime
    
    // Page should load within reasonable time (5 seconds for dev environment)
    expect(loadTime).toBeLessThan(5000)
  })

  test('pages are accessible via keyboard navigation', async ({ page }) => {
    await page.goto('/')
    
    // Test tab navigation
    await page.keyboard.press('Tab')
    
    // Should be able to focus on interactive elements
    const focusedElement = await page.locator(':focus')
    if (await focusedElement.count() > 0) {
      await expect(focusedElement).toBeVisible()
    }
  })

  test('application handles network errors gracefully', async ({ page }) => {
    await page.goto('/')
    
    // Simulate offline condition
    await page.context().setOffline(true)
    
    // Try to interact with the page
    const buttons = page.locator('button').first()
    if (await buttons.count() > 0) {
      await buttons.click()
      
      // Page should still be responsive (not frozen)
      await expect(page.locator('body')).toBeVisible()
    }
    
    // Restore online condition
    await page.context().setOffline(false)
  })

  test('memory usage is reasonable', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    // Get initial memory usage
    const metrics = await page.evaluate(() => {
      const memory = (performance as any).memory
      return memory ? {
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit
      } : null
    })
    
    if (metrics) {
      // Used memory should be reasonable (less than 100MB for basic page)
      expect(metrics.usedJSHeapSize).toBeLessThan(100 * 1024 * 1024)
    }
  })

  test('large data sets render without blocking UI', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForLoadState('networkidle')
    
    // Test that UI remains responsive during data loading
    const startTime = Date.now()
    
    // Try to scroll the page
    await page.mouse.wheel(0, 500)
    await page.waitForTimeout(100)
    
    // Scrolling should be responsive (quick)
    const scrollTime = Date.now() - startTime
    expect(scrollTime).toBeLessThan(1000)
    
    // Page should still be interactive
    await expect(page.locator('body')).toBeVisible()
  })
})