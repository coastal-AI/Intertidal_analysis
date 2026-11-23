/**
 * Light and dark themes. The inline script of index.html applies the theme before the first paint with the same key
 * and rule: the visitor's stored choice, else the system preference. Storage may be unavailable (private windows,
 * blocked site data), so every access is wrapped in try/catch and the site works without it.
 */
import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

export const THEME_STORAGE_KEY = 'marea-theme'
const THEME_COLOR: Record<Theme, string> = { light: '#F2F6F7', dark: '#0D161A' }

export function storedTheme(): Theme | null {
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY)
    return v === 'light' || v === 'dark' ? v : null
  } catch {
    return null
  }
}

function storeTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    // Storage blocked: the choice lasts for this page view only.
  }
}

export function systemTheme(): Theme {
  return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}

/** The theme on screen now (the class on <html>). */
export function currentTheme(): Theme {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement
  root.classList.toggle('dark', theme === 'dark')
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', THEME_COLOR[theme])
  window.dispatchEvent(new CustomEvent<Theme>('marea-theme', { detail: theme }))
}

/**
 * The current theme and a setter that applies and stores it. Without a stored choice the theme follows the system
 * preference live. Components that draw with theme colours (maps, charts) can listen to the "marea-theme" window
 * event or use this hook.
 */
export function useTheme(): [Theme, (theme: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>(() => currentTheme())

  useEffect(() => {
    const onTheme = (e: Event) => setThemeState((e as CustomEvent<Theme>).detail)
    window.addEventListener('marea-theme', onTheme)
    const mq = typeof window.matchMedia === 'function' ? window.matchMedia('(prefers-color-scheme: dark)') : null
    const onSystem = () => {
      if (storedTheme() === null) applyTheme(systemTheme())
    }
    mq?.addEventListener('change', onSystem)
    return () => {
      window.removeEventListener('marea-theme', onTheme)
      mq?.removeEventListener('change', onSystem)
    }
  }, [])

  const setTheme = useCallback((next: Theme) => {
    storeTheme(next)
    applyTheme(next)
  }, [])

  return [theme, setTheme]
}
