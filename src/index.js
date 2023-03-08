import React from 'react';
import Chessboard from './Chessboard';
import './index.css';
import { createRoot } from 'react-dom/client';

const domNode = document.getElementById('root');
const root = createRoot(domNode);

root.render(
  <React.StrictMode>
    <Chessboard />
  </React.StrictMode>
);
