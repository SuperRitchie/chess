import React from 'react';
import './ChessPiece.css';

// Create a ChessPiece component
// svg chess pieces from https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces


function ChessPiece({ piece }) {
    const pieceName = piece ? `${piece.color}-${piece.type}` : '';
    const pieceSrc = piece ? `assets/${pieceName}.svg` : '';

    console.log(pieceSrc);
    console.log(pieceName);


    return (
        <div className="chess-piece">
            {piece && <img src={pieceSrc} alt={pieceName} />}
        </div>
    );
}

export default ChessPiece;
