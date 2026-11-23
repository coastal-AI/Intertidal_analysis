import { describe, expect, it } from 'vitest'
import { EN_DASH, MINUS, NBSP, fmtInt, fmtMinutes, fmtNumber, fmtPercent } from './format'

describe('numbers', () => {
  it('writes negatives with the real minus sign', () => {
    expect(fmtNumber(-0.46)).toBe(`${MINUS}0.46`)
  })
  it('writes "+" only when asked', () => {
    expect(fmtNumber(0.1)).toBe('0.10')
    expect(fmtNumber(0.1, { sign: true })).toBe('+0.10')
  })
  it('never writes a signed zero', () => {
    expect(fmtNumber(-0.001)).toBe('0.00')
    expect(fmtNumber(0.001, { sign: true })).toBe('0.00')
  })
  it('groups thousands and rounds', () => {
    expect(fmtNumber(29728, { decimals: 0 })).toBe('29,728')
    expect(fmtInt(714.6)).toBe('715')
  })
  it('shows an en dash for missing values', () => {
    expect(fmtNumber(Number.NaN)).toBe(EN_DASH)
    expect(fmtInt(Number.NaN)).toBe(EN_DASH)
  })
})

describe('units', () => {
  it('puts a no-break space before the unit', () => {
    expect(NBSP).toBe(' ')
    expect(fmtPercent(42.24)).toBe(`42${NBSP}%`)
    expect(fmtMinutes(7)).toBe(`7${NBSP}min`)
  })
})
