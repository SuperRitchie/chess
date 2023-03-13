import React from 'react';
import './ChessPiece.css';
import { ReactSVG } from 'react-svg';


// Create a ChessPiece component
// svg chess pieces from https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces


function ChessPiece({ piece }) {
    const pieceName = piece ? `${piece.color}-${piece.type}` : '';
    const pieceSrc = piece ? `${pieceName}.svg` : '';
    return (
        <div className="chess-piece">
            {piece && <ReactSVG src={pieceSrc} alt={pieceName} />}
        </div>
    );
}

export default ChessPiece;
