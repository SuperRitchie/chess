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

  const handleSquareClick = (x, y) => {
    if (selectedSquare && selectedSquare.x === x && selectedSquare.y === y) {
      setSelectedSquare(null);
    } else {
      setSelectedSquare({ x, y });
    }
  };

  const handleOnDrag = (e, x, y) => {
    const piece = getPiece(pieces, x, y);
    if (!piece) { e.preventDefault(); return; }
    // Only allow dragging the side to move
    if ((isWhiteTurn && piece.color !== 'white') || (!isWhiteTurn && piece.color !== 'black')) {
      e.preventDefault();
      return;
    }
    e.dataTransfer.setData('position', `${x}-${y}`);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleOnDrop = (e, x, y) => {
    const [oldX, oldY] = e.dataTransfer.getData('position').split('-').map(Number);
    const from = { x: oldX, y: oldY };
    const to = { x, y };

    if (!isLegalMove(pieces, from, to, isWhiteTurn)) {
        playSound(check);
      // add error sound
      return;
    }

    const newPieces = makeMove(pieces, from, to);
    setPieces(newPieces);
    setIsWhiteTurn((t) => !t);
    playSound(move_sound);

    // check if player is in check --> if (isCheck(newPieces, isWhiteTurn ? 'black' : 'white')) playSound(check);
  };

  const playSound = (audioFile) => {
    const audio = new Audio(audioFile);
    audio.play();
  };

  return (
    <div className="chessboard">
      {Array.from({ length: 8 }, (_, y) => (
        <div key={y} className="chessboard-row">
          {Array.from({ length: 8 }, (_, x) => {
            const piece = getPiece(pieces, x, y);
            const selected = selectedSquare && selectedSquare.x === x && selectedSquare.y === y;
            return (
              <div
                key={`${x}-${y}`}
                className={`chessboard-square ${isBlackSquare(x, y) ? 'chessboard-square--black' : 'chessboard-square--white'} ${selected ? 'chessboard-square--selected' : ''}`}
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
