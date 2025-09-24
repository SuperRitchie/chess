// ChessBoard.jsx
import React, { useState } from 'react';
import './index.css';
import ChessPiece from './ChessPiece';
import move_sound from './assets/move.mp3';
import check from './assets/check.mp3';
import { isLegalMove, makeMove, getPiece } from './chessRules';

const ChessBoard = () => {
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [isWhiteTurn, setIsWhiteTurn] = useState(true);
  const [legalMoves, setLegalMoves] = useState([]); // [{x,y,capture:boolean}]

  const [pieces, setPieces] = useState({
    '0-0': { color: 'black', type: 'rook' },
    '0-1': { color: 'black', type: 'knight' },
    '0-2': { color: 'black', type: 'bishop' },
    '0-3': { color: 'black', type: 'queen' },
    '0-4': { color: 'black', type: 'king' },
    '0-5': { color: 'black', type: 'bishop' },
    '0-6': { color: 'black', type: 'knight' },
    '0-7': { color: 'black', type: 'rook' },
    '1-0': { color: 'black', type: 'pawn' },
    '1-1': { color: 'black', type: 'pawn' },
    '1-2': { color: 'black', type: 'pawn' },
    '1-3': { color: 'black', type: 'pawn' },
    '1-4': { color: 'black', type: 'pawn' },
    '1-5': { color: 'black', type: 'pawn' },
    '1-6': { color: 'black', type: 'pawn' },
    '1-7': { color: 'black', type: 'pawn' },
    '6-0': { color: 'white', type: 'pawn' },
    '6-1': { color: 'white', type: 'pawn' },
    '6-2': { color: 'white', type: 'pawn' },
    '6-3': { color: 'white', type: 'pawn' },
    '6-4': { color: 'white', type: 'pawn' },
    '6-5': { color: 'white', type: 'pawn' },
    '6-6': { color: 'white', type: 'pawn' },
    '6-7': { color: 'white', type: 'pawn' },
    '7-0': { color: 'white', type: 'rook' },
    '7-1': { color: 'white', type: 'knight' },
    '7-2': { color: 'white', type: 'bishop' },
    '7-3': { color: 'white', type: 'queen' },
    '7-4': { color: 'white', type: 'king' },
    '7-5': { color: 'white', type: 'bishop' },
    '7-6': { color: 'white', type: 'knight' },
    '7-7': { color: 'white', type: 'rook' },
  });

  const isBlackSquare = (x, y) => ((x + y) % 2 === 1);

  const computeLegalMoves = (fromX, fromY) => {
    const res = [];
    for (let ty = 0; ty < 8; ty++) {
      for (let tx = 0; tx < 8; tx++) {
        const from = { x: fromX, y: fromY };
        const to = { x: tx, y: ty };
        if (isLegalMove(pieces, from, to, isWhiteTurn)) {
          const targetPiece = getPiece(pieces, tx, ty);
          res.push({ x: tx, y: ty, capture: !!targetPiece });
        }
      }
    }
    return res;
  };

  const clearSelection = () => {
    setSelectedSquare(null);
    setLegalMoves([]);
  };

  const handleSquareClick = (x, y) => {
    const piece = getPiece(pieces, x, y);

    // Clicking the selected square toggles off
    if (selectedSquare && selectedSquare.x === x && selectedSquare.y === y) {
      clearSelection();
      return;
    }

    // If you click a piece of the side to move: select and show moves
    if (piece && ((isWhiteTurn && piece.color === 'white') || (!isWhiteTurn && piece.color === 'black'))) {
      setSelectedSquare({ x, y });
      setLegalMoves(computeLegalMoves(x, y));
      return;
    }

    // Otherwise (empty square or opponent piece) just update selection if a move isn't being executed by click.
    // (Drag/drop still handles the actual move.)
    setSelectedSquare(null);
    setLegalMoves([]);
  };

  const handleOnDrag = (e, x, y) => {
    const piece = getPiece(pieces, x, y);
    if (!piece) { e.preventDefault(); return; }
    if ((isWhiteTurn && piece.color !== 'white') || (!isWhiteTurn && piece.color !== 'black')) {
      e.preventDefault();
      return;
    }
    // show moves on drag start too (nice UX)
    setSelectedSquare({ x, y });
    setLegalMoves(computeLegalMoves(x, y));

    e.dataTransfer.setData('position', `${x}-${y}`);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleOnDrop = (e, x, y) => {
    const data = e.dataTransfer.getData('position');
    if (!data) return;
    const [oldX, oldY] = data.split('-').map(Number);
    const from = { x: oldX, y: oldY };
    const to = { x, y };

    if (!isLegalMove(pieces, from, to, isWhiteTurn)) {
      playSound(check); // error sound for illegal attempt
      return;
    }

    const newPieces = makeMove(pieces, from, to);
    setPieces(newPieces);
    setIsWhiteTurn((t) => !t);
    playSound(move_sound);
    clearSelection();
  };

  const playSound = (audioFile) => {
    const audio = new Audio(audioFile);
    audio.play();
  };

  // Helper to see if a square is a legal target (and if it's a capture)
  const legalForSquare = (x, y) => legalMoves.find(m => m.x === x && m.y === y);

  return (
    <div className="chessboard">
      {Array.from({ length: 8 }, (_, y) => (
        <div key={y} className="chessboard-row">
          {Array.from({ length: 8 }, (_, x) => {
            const piece = getPiece(pieces, x, y);
            const selected = selectedSquare && selectedSquare.x === x && selectedSquare.y === y;
            const lm = legalForSquare(x, y);
            const isCapture = lm?.capture;

            return (
              <div
                key={`${x}-${y}`}
                className={[
                  'chessboard-square',
                  isBlackSquare(x, y) ? 'chessboard-square--black' : 'chessboard-square--white',
                  selected ? 'chessboard-square--selected' : '',
                  lm ? (isCapture ? 'chessboard-square--legal-capture' : 'chessboard-square--legal-move') : ''
                ].join(' ')}
                onClick={() => handleSquareClick(x, y)}
                onDrop={(e) => handleOnDrop(e, x, y)}
                onDragOver={handleDragOver}
              >
                <div
                  className="chesspiece-wrapper"
                  draggable={!!piece && ((isWhiteTurn && piece.color === 'white') || (!isWhiteTurn && piece.color === 'black'))}
                  onDragStart={(e) => handleOnDrag(e, x, y)}
                >
                  <ChessPiece piece={piece} />
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

export default ChessBoard;
