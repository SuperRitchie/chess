import React from 'react';
import ChessBoard from './ChessBoard';
import './index.css';
import { createRoot } from 'react-dom/client';

const domNode = document.getElementById('root');
const root = createRoot(domNode);

root.render(
  <React.StrictMode>
    <ChessBoard />
  </React.StrictMode>
);
