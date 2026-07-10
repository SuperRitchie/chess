import { moveToPolicyIndex, POLICY_SIZE } from './nnAI';

describe('neural policy encoding', () => {
  test('promotion choices have distinct policy indices', () => {
    const base = {
      from: { x: 1, y: 0 },
      to: { x: 0, y: 0 },
      needsPromotion: true,
    };
    const indices = ['queen', 'rook', 'bishop', 'knight'].map((promotionType) =>
      moveToPolicyIndex({ ...base, promotionType }),
    );

    expect(new Set(indices).size).toBe(4);
    indices.forEach((index) => expect(index).toBeGreaterThanOrEqual(0));
    indices.forEach((index) => expect(index).toBeLessThan(POLICY_SIZE));
  });

  test('legacy policy mapping remains available during checkpoint migration', () => {
    const move = {
      from: { x: 6, y: 4 },
      to: { x: 4, y: 4 },
    };
    expect(moveToPolicyIndex(move, 4096)).toBe(12 * 64 + 28);
  });
});
