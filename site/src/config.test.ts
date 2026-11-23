import { describe, expect, it } from 'vitest'
import { DEFAULT_REPO_URL, parseEnable3d, parseRepoUrl, repoFile } from './config'

describe('VITE_ENABLE_3D', () => {
  it('defaults to "desktop"', () => {
    expect(parseEnable3d(undefined)).toBe('desktop')
    expect(parseEnable3d('')).toBe('desktop')
    expect(parseEnable3d('maybe')).toBe('desktop')
  })
  it('reads "on" and "off"', () => {
    expect(parseEnable3d('on')).toBe('on')
    expect(parseEnable3d(' OFF ')).toBe('off')
  })
})

describe('repository links', () => {
  it('parses the default branch URL', () => {
    expect(parseRepoUrl(DEFAULT_REPO_URL)).toEqual({ owner: 'coastal-AI', name: 'Intertidal_analysis', ref: 'CARTOGRASS26' })
  })
  it('links files at the branch', () => {
    expect(repoFile('LICENSE', DEFAULT_REPO_URL)).toBe('https://github.com/coastal-AI/Intertidal_analysis/blob/CARTOGRASS26/LICENSE')
  })
  it('handles a repository URL without a branch, and other hosts', () => {
    expect(parseRepoUrl('https://github.com/owner/marea-site')).toEqual({ owner: 'owner', name: 'marea-site', ref: '' })
    expect(repoFile('README.md', 'https://github.com/owner/marea-site')).toBe('https://github.com/owner/marea-site/blob/HEAD/README.md')
    expect(parseRepoUrl('https://example.org/marea')).toBeNull()
    expect(repoFile('README.md', 'https://example.org/marea')).toBe('https://example.org/marea/README.md')
  })
})
