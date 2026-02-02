/**
 * Visualization Canvas Component
 * Renders the real-time visualization of the training environment
 */

import React, { useRef, useEffect, useCallback } from 'react';
import { useStore } from '../../store/useStore';

interface VisualizationCanvasProps {
  width?: number;
  height?: number;
  className?: string;
}

export const VisualizationCanvas: React.FC<VisualizationCanvasProps> = ({
  width = 800,
  height = 600,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  const {
    visualization,
    environment,
    isTraining,
    isPlaying,
    playbackSpeed,
  } = useStore();

  /**
   * Draw the environment background
   */
  const drawBackground = useCallback((ctx: CanvasRenderingContext2D) => {
    // Dark gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, '#0a0a0f');
    gradient.addColorStop(1, '#141420');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Grid pattern
    ctx.strokeStyle = 'rgba(100, 255, 218, 0.05)';
    ctx.lineWidth = 1;

    const gridSize = 40;
    for (let x = 0; x <= width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y <= height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }, [width, height]);

  /**
   * Draw obstacles
   */
  const drawObstacles = useCallback((ctx: CanvasRenderingContext2D) => {
    const obstacles = visualization.obstacles || environment.obstacles || [];

    obstacles.forEach((obstacle) => {
      const x = obstacle.x || obstacle.position?.[0] || 0;
      const y = obstacle.y || obstacle.position?.[1] || 0;
      const w = obstacle.width || obstacle.size?.[0] || 20;
      const h = obstacle.height || obstacle.size?.[1] || 20;

      // Obstacle glow
      ctx.shadowColor = '#ff6b6b';
      ctx.shadowBlur = 10;

      // Obstacle body
      ctx.fillStyle = 'rgba(255, 107, 107, 0.8)';
      ctx.fillRect(x, y, w, h);

      // Obstacle border
      ctx.strokeStyle = '#ff6b6b';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      ctx.shadowBlur = 0;
    });
  }, [visualization.obstacles, environment.obstacles]);

  /**
   * Draw the goal
   */
  const drawGoal = useCallback((ctx: CanvasRenderingContext2D) => {
    const goalPos = visualization.goalPosition || environment.goals?.[0]?.position || [700, 300];
    const [gx, gy] = goalPos;
    const goalSize = 40;

    // Goal glow
    ctx.shadowColor = '#4ecdc4';
    ctx.shadowBlur = 20;

    // Pulsing effect
    const pulse = Math.sin(Date.now() / 200) * 5 + goalSize;

    // Goal circle
    ctx.beginPath();
    ctx.arc(gx, gy, pulse / 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(78, 205, 196, 0.3)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(gx, gy, goalSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#4ecdc4';
    ctx.fill();

    // Goal inner circle
    ctx.beginPath();
    ctx.arc(gx, gy, goalSize / 4, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();

    ctx.shadowBlur = 0;
  }, [visualization.goalPosition, environment.goals]);

  /**
   * Draw the agent trajectory
   */
  const drawTrajectory = useCallback((ctx: CanvasRenderingContext2D) => {
    const trajectory = visualization.trajectory || [];

    if (trajectory.length < 2) return;

    ctx.strokeStyle = 'rgba(100, 255, 218, 0.3)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(trajectory[0][0], trajectory[0][1]);

    for (let i = 1; i < trajectory.length; i++) {
      const opacity = (i / trajectory.length) * 0.5 + 0.1;
      ctx.strokeStyle = `rgba(100, 255, 218, ${opacity})`;
      ctx.lineTo(trajectory[i][0], trajectory[i][1]);
    }

    ctx.stroke();
  }, [visualization.trajectory]);

  /**
   * Draw the agent
   */
  const drawAgent = useCallback((ctx: CanvasRenderingContext2D) => {
    const [ax, ay] = visualization.agentPosition || environment.agentPosition || [100, 300];
    const agentSize = 24;

    // Agent glow
    ctx.shadowColor = '#64ffda';
    ctx.shadowBlur = 15;

    // Agent body (triangle pointing right by default)
    ctx.fillStyle = '#64ffda';
    ctx.beginPath();
    ctx.moveTo(ax + agentSize, ay);
    ctx.lineTo(ax - agentSize / 2, ay - agentSize / 2);
    ctx.lineTo(ax - agentSize / 2, ay + agentSize / 2);
    ctx.closePath();
    ctx.fill();

    // Agent core
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(ax, ay, agentSize / 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowBlur = 0;
  }, [visualization.agentPosition, environment.agentPosition]);

  /**
   * Draw status overlay
   */
  const drawStatus = useCallback((ctx: CanvasRenderingContext2D) => {
    // Status indicator
    ctx.font = '14px "JetBrains Mono", monospace';
    ctx.fillStyle = isTraining ? '#64ffda' : '#888';
    ctx.fillText(
      isTraining ? (isPlaying ? '● TRAINING' : '◐ PAUSED') : '○ IDLE',
      20,
      30
    );

    // Speed indicator
    if (isTraining) {
      ctx.fillStyle = '#888';
      ctx.fillText(`Speed: ${playbackSpeed}x`, 20, 50);
    }
  }, [isTraining, isPlaying, playbackSpeed]);

  /**
   * Main render function
   */
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear and draw
    ctx.clearRect(0, 0, width, height);

    drawBackground(ctx);
    drawObstacles(ctx);
    drawTrajectory(ctx);
    drawGoal(ctx);
    drawAgent(ctx);
    drawStatus(ctx);

    // Continue animation loop
    if (isTraining && isPlaying) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [
    width,
    height,
    drawBackground,
    drawObstacles,
    drawTrajectory,
    drawGoal,
    drawAgent,
    drawStatus,
    isTraining,
    isPlaying,
  ]);

  // Start/stop animation loop
  useEffect(() => {
    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  // Re-render on visualization state changes
  useEffect(() => {
    render();
  }, [visualization, environment, render]);

  return (
    <div className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded-lg border border-primary/20"
        style={{
          background: 'linear-gradient(180deg, #0a0a0f 0%, #141420 100%)',
        }}
      />

      {/* Overlay controls */}
      <div className="absolute bottom-4 right-4 flex gap-2">
        <div className="bg-surface/80 backdrop-blur-sm px-3 py-1 rounded-full text-xs text-secondary border border-primary/20">
          {width} × {height}
        </div>
      </div>

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-surface/80 backdrop-blur-sm p-3 rounded-lg border border-primary/20">
        <div className="text-xs space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-primary rounded-full" />
            <span className="text-secondary">Agent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-accent rounded-full" />
            <span className="text-secondary">Goal</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-danger rounded" />
            <span className="text-secondary">Obstacle</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizationCanvas;
