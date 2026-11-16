import { fireEvent, render, screen } from '@testing-library/react';
import ChessBoard from './ChessBoard';


const squareIndex = (square) => {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]);
  return (7 - file) * 8 + (8 - rank);
};


describe('chess board game status', () => {
  beforeEach(() => {
    global.Audio = class {
      play() {}
    };
  });

  test('reports checkmate in the page without a blocking dialog', () => {
    const { container } = render(<ChessBoard />);
    const squares = container.querySelectorAll('.chessboard-square');
    const play = (uci) => {
      fireEvent.click(squares[squareIndex(uci.slice(0, 2))]);
      fireEvent.click(squares[squareIndex(uci.slice(2, 4))]);
    };

    ['e2e4', 'e7e5', 'f1c4', 'b8c6', 'd1h5', 'g8f6', 'h5f7'].forEach(play);

    expect(screen.getByRole('status').textContent).toBe('black is checkmated');
  });
});
