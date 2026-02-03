import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, FileText, Download, CheckCircle, AlertCircle, Eye, Activity, Database, Play, Settings, RefreshCw, HelpCircle, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';

/**
 * MarkSheet Reader Application
 * * 概要:
 * - OpenCV.jsを使用したクライアントサイド画像処理
 * - ArUcoマーカー(または矩形)を検出し、射影変換で歪み補正
 * - グリッドベースのOMR (Optical Mark Recognition)
 * - 階層的行動データのロジック適用
 * * データ構造 (修正版):
 * - 1シート = 4ブロック(4人)
 * - 1ブロック = 6カラム [No., 男性, 女性, Look(気づいた), Stop(立ち止まった), Use(使った)]
 * - 行数 = 61行 (ヘッダー1行 + データ60行)
 * - ID順序: ブロックごとに縦優先 (Block 1: 1-60, Block 2: 61-120...)
 */

// 定数定義
const GRID_ROWS = 61; // インデックス行含む
const GRID_COLS = 24; // 6カラム * 4人
const BLOCKS_PER_ROW = 4;
const COLS_PER_BLOCK = 6;
const HEADER_ROWS_TO_SKIP = 1;
const DATA_ROWS_PER_BLOCK = 60; // 1ブロックあたりのデータ行数

// データ項目の定義（デバッグ表示用）
const DATA_LABELS = ['No.', 'Male', 'Fem', 'Look', 'Stop', 'Use'];

const App = () => {
  const [cvLoaded, setCvLoaded] = useState(false);
  const [images, setImages] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [results, setResults] = useState([]);
  const [debugData, setDebugData] = useState(null); // { warpedImageURL, cells: [], errorImage: null }
  const [logs, setLogs] = useState([]);
  const [showGuide, setShowGuide] = useState(false);
  const [isDragging, setIsDragging] = useState(false); // ドラッグ状態管理
  
  // しきい値をStateで管理 (デフォルト0.2)
  const [bwThreshold, setBwThreshold] = useState(0.2);

  // ロード状態を管理するRef (StrictMode対策)
  const isCvLoading = useRef(false);

  // OpenCVのロード
  useEffect(() => {
    // 既にロード済み、またはロード中なら何もしない（二重読み込み防止）
    if (window.cv && window.cv.Mat) {
        if (!cvLoaded) {
            setCvLoaded(true);
            addLog('OpenCV.js is already ready! ✨');
        }
        return;
    }
    if (isCvLoading.current) return;

    // DOM上にスクリプトタグが既にあるかチェック
    if (document.querySelector('script[src="https://docs.opencv.org/4.8.0/opencv.js"]')) {
        return;
    }

    isCvLoading.current = true;
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.8.0/opencv.js';
    script.async = true;
    
    script.onload = () => {
      // OpenCV.jsのロード完了後、Wasmの初期化を待つ必要がある
      // cv.onRuntimeInitialized が提供されている場合はそれを待つ
      if (window.cv) {
          window.cv.onRuntimeInitialized = () => {
              setCvLoaded(true);
              addLog('OpenCV.js runtime initialized! Ready to roll! ✨');
              isCvLoading.current = false;
          };

          // すでに初期化済みの場合のフォールバック
          if (window.cv.Mat) {
             setCvLoaded(true);
             addLog('OpenCV.js loaded successfully. ✨');
             isCvLoading.current = false;
          }
      }
    };

    script.onerror = () => {
      addLog('Failed to load OpenCV.js. Please refresh.', 'error');
      isCvLoading.current = false;
    };

    document.body.appendChild(script);

    // アンマウント時にスクリプトを削除しない（再マウント時のコスト回避）
  }, []);

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev, { msg, type, time: new Date().toLocaleTimeString() }]);
  };

  const handleImageUpload = (e) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files).sort((a, b) => a.name.localeCompare(b.name));
    setImages(files);
    setResults([]);
    setDebugData(null);
    addLog(`Uploaded ${files.length} images. Sorted by filename.`);
  };

  // ドラッグ＆ドロップ用ハンドラ
  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (!cvLoaded) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      // 画像ファイルのみフィルタリング
      const files = Array.from(e.dataTransfer.files)
        .filter(file => file.type.startsWith('image/'))
        .sort((a, b) => a.name.localeCompare(b.name));
      
      if (files.length > 0) {
        setImages(files);
        setResults([]);
        setDebugData(null);
        addLog(`Uploaded ${files.length} images via Drag & Drop.`);
      } else {
        addLog('No valid image files found.', 'error');
      }
    }
  };

  /**
   * 画像処理のメインロジック
   */
  const processImages = async () => {
    if (!cvLoaded || images.length === 0) return;

    setProcessing(true);
    setResults([]);
    setProgress({ current: 0, total: images.length });
    const newResults = [];

    // cvオブジェクトへのショートカット
    const cv = window.cv;

    for (let i = 0; i < images.length; i++) {
      const file = images[i];
      setProgress({ current: i + 1, total: images.length });
      addLog(`Processing: ${file.name}...`);

      try {
        const imgData = await loadImage(file);
        // 1枚目か、エラー発生時にデバッグデータを生成するように変更
        // 現在のしきい値を渡す
        const result = processSingleImage(cv, imgData, file.name, true, bwThreshold); 
        
        if (result.success) {
          newResults.push(result.data);
          // 成功時、最初の1枚目ならデバッグ表示
          if (i === 0) {
            setDebugData(result.debug);
          }
          addLog(`Processed ${file.name}: Detected ${result.data.peopleCount} people.`);
        } else {
          // 失敗時、デバッグデータをセットしてユーザーに見せる
          setDebugData(result.debug);
          addLog(`Error processing ${file.name}: ${result.error}`, 'error');
        }
      } catch (err) {
        console.error(err);
        addLog(`Exception on ${file.name}: ${err.message}`, 'error');
      }
    }

    setResults(newResults);
    setProcessing(false);
    addLog('Processing completed.');
  };

  // ファイルから画像データを読み込む
  const loadImage = (file) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  };

  /**
   * 単一画像の処理パイプライン
   * 1. 画像読み込み & パディング追加
   * 2. 二値化 & マーカー検出 (4隅)
   * 3. 射影変換 (歪み補正)
   * 4. グリッド分割 & OMR
   * 5. データ構造化
   */
  const processSingleImage = (cv, imgElement, filename, generateDebug = false, thresholdVal) => {
    let srcOriginal = null;
    let src = null;
    let dst = null;
    let gray = null;
    let thresholded = null;
    let contours = null;
    let hierarchy = null;
    let debugSrc = null;
    let srcTri = null;
    let dstTri = null;
    let warpedGray = null;
    let warpedThresh = null;
    let markers = []; // マーカー配列（approxのメモリ管理用）

    try {
      srcOriginal = cv.imread(imgElement);
      src = new cv.Mat();
      dst = new cv.Mat();
      gray = new cv.Mat();
      thresholded = new cv.Mat();
      contours = new cv.MatVector();
      hierarchy = new cv.Mat();

      // 1. パディング追加 (画像の端にあるマーカー対策)
      cv.copyMakeBorder(srcOriginal, src, 20, 20, 20, 20, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));
      srcOriginal.delete(); srcOriginal = null; // 早めに解放

      // 前処理
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      
      // 二値化
      cv.adaptiveThreshold(gray, thresholded, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10);
      
      if (generateDebug) {
          debugSrc = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
          cv.cvtColor(thresholded, debugSrc, cv.COLOR_GRAY2RGB);
      }

      // 2. マーカー候補検出
      cv.findContours(thresholded, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

      const totalPixels = src.cols * src.rows;
      const minArea = totalPixels * 0.0005; 
      const maxArea = totalPixels * 0.3; 
      
      let countQuad = 0;
      let countSizeOK = 0;

      for (let i = 0; i < contours.size(); i++) {
        let cnt = contours.get(i);
        let area = cv.contourArea(cnt);
        
        if (area > minArea && area < maxArea) {
          countSizeOK++;
          let peri = cv.arcLength(cnt, true);
          let approx = new cv.Mat();
          cv.approxPolyDP(cnt, approx, 0.05 * peri, true); 

          if (approx.rows === 4 && cv.isContourConvex(approx)) {
             countQuad++;
             // 重心を計算
             let M = cv.moments(approx);
             let cx = M.m10 / M.m00;
             let cy = M.m01 / M.m00;
             markers.push({ cx, cy, approx: approx }); // approxを保存(後でdeleteが必要)

             if (generateDebug && debugSrc) {
                 let pts = new cv.MatVector();
                 pts.push_back(approx);
                 cv.drawContours(debugSrc, pts, 0, new cv.Scalar(0, 255, 0, 255), 4);
                 pts.delete();
             }
          } else {
             approx.delete(); // 使わないなら即解放
          }
        }
      }

      const createDebugImage = (mat, showGrid = false) => {
          // グリッド線を描画するためのコピーを作成
          let displayMat = mat.clone();
          
          if (showGrid) {
            const width = displayMat.cols;
            const height = displayMat.rows;
            const cellW = width / GRID_COLS;
            const cellH = height / GRID_ROWS;

            // 水色の線を引く
            const lineColor = new cv.Scalar(0, 255, 255, 255); // RGBA (Cyan)

            // 横線 (Rows)
            for (let r = 0; r <= GRID_ROWS; r++) {
               let y = Math.floor(r * cellH);
               cv.line(displayMat, new cv.Point(0, y), new cv.Point(width, y), lineColor, 2);
            }
            // 縦線 (Cols)
            for (let c = 0; c <= GRID_COLS; c++) {
               let x = Math.floor(c * cellW);
               cv.line(displayMat, new cv.Point(x, 0), new cv.Point(x, height), lineColor, 2);
            }
          }

          const canvas = document.createElement('canvas');
          cv.imshow(canvas, displayMat);
          displayMat.delete(); // コピーを解放
          return canvas.toDataURL();
      };

      if (markers.length < 4) {
        let errorMsg = `Found ${markers.length} markers. (Total Contours: ${contours.size()}, SizeOK: ${countSizeOK}, Quads: ${countQuad})`;
        let errorImgURL = null;
        if (generateDebug && debugSrc) {
            errorImgURL = createDebugImage(debugSrc, false);
        }
        // マーカーのメモリ解放
        markers.forEach(m => m.approx.delete());
        return { success: false, error: errorMsg, debug: { image: errorImgURL, cells: [], isError: true } };
      }

      // マーカーソート
      markers.sort((a, b) => cv.contourArea(b.approx) - cv.contourArea(a.approx));
      
      // 上位4つ以外は解放
      for(let i = 4; i < markers.length; i++) {
         markers[i].approx.delete();
      }
      markers = markers.slice(0, 4);

      markers.sort((a, b) => a.cy - b.cy);
      const topMarkers = markers.slice(0, 2).sort((a, b) => a.cx - b.cx);
      const bottomMarkers = markers.slice(2, 4).sort((a, b) => a.cx - b.cx);
      const sortedMarkers = [topMarkers[0], topMarkers[1], bottomMarkers[1], bottomMarkers[0]];

      // 3. マーカー内側取得
      const srcCoords = [];
      const imgCenter = { x: src.cols / 2, y: src.rows / 2 };

      sortedMarkers.forEach((marker) => {
        let bestPoint = null;
        let minDist = Infinity;
        const data = marker.approx.data32S;

        for (let j = 0; j < 4; j++) {
          const px = data[j * 2];
          const py = data[j * 2 + 1];
          const dist = Math.hypot(px - imgCenter.x, py - imgCenter.y);
          if (dist < minDist) {
            minDist = dist;
            bestPoint = { x: px, y: py };
          }
        }
        srcCoords.push(bestPoint.x);
        srcCoords.push(bestPoint.y);
        
        if (generateDebug && debugSrc) {
            cv.circle(debugSrc, new cv.Point(bestPoint.x, bestPoint.y), 10, new cv.Scalar(255, 0, 0, 255), -1);
        }
      });

      // 4. 射影変換
      const outWidth = 1200;
      const outHeight = 3050;
      
      srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, srcCoords);
      dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, outWidth, 0, outWidth, outHeight, 0, outHeight]);
      
      let M = cv.getPerspectiveTransform(srcTri, dstTri);
      cv.warpPerspective(src, dst, M, new cv.Size(outWidth, outHeight), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
      M.delete(); // 行列は即解放

      // 5. グリッド処理 & OMR
      warpedGray = new cv.Mat();
      warpedThresh = new cv.Mat();
      cv.cvtColor(dst, warpedGray, cv.COLOR_RGBA2GRAY);
      cv.threshold(warpedGray, warpedThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

      const cellW = outWidth / GRID_COLS;
      const cellH = outHeight / GRID_ROWS;
      
      const sheetData = [];
      const debugCells = [];

      for (let r = HEADER_ROWS_TO_SKIP; r < GRID_ROWS; r++) {
        for (let blockIdx = 0; blockIdx < BLOCKS_PER_ROW; blockIdx++) {
          
          // ID計算
          const dataRowIndex = r - HEADER_ROWS_TO_SKIP + 1; // 1 ~ 60
          const personId = blockIdx * DATA_ROWS_PER_BLOCK + dataRowIndex;

          const personData = {
            id: personId,
            raw: [],
            logic: {}
          };
          
          for (let c = 0; c < COLS_PER_BLOCK; c++) {
            const globalCol = blockIdx * COLS_PER_BLOCK + c;
            const marginX = cellW * 0.25;
            const marginY = cellH * 0.25;
            const x = Math.floor(globalCol * cellW + marginX);
            const y = Math.floor(r * cellH + marginY);
            const w = Math.floor(cellW - marginX * 2);
            const h = Math.floor(cellH - marginY * 2);

            let roi = warpedThresh.roi(new cv.Rect(x, y, w, h));
            let nonZero = cv.countNonZero(roi);
            let totalPixels = w * h;
            let ratio = nonZero / totalPixels;
            
            // デバッグ用: 最初の行のセル画像を保存
            if (generateDebug && r === HEADER_ROWS_TO_SKIP) {
              let roiDisplay = dst.roi(new cv.Rect(x, y, w, h));
              const canvas = document.createElement('canvas');
              cv.imshow(canvas, roiDisplay);
              const cellImgUrl = canvas.toDataURL();
              
              debugCells.push({ 
                label: DATA_LABELS[c],
                person: personId, // IDを表示
                image: cellImgUrl, 
                ratio: ratio.toFixed(2), 
                checked: ratio > thresholdVal 
              });
              
              roiDisplay.delete();
            }

            roi.delete();
            personData.raw.push(ratio > thresholdVal ? 1 : 0);
          }
          
          const [_no, male, female, lookRaw, stopRaw, useRaw] = personData.raw;
          const isActive = (male || female);

          if (isActive) {
            // 階層ロジック
            const logicUsed = useRaw === 1;
            const logicStopped = logicUsed || stopRaw === 1;
            const logicNoticed = logicStopped || lookRaw === 1;

            personData.logic = {
              male: male,
              female: female,
              noticed: logicNoticed ? 1 : 0,
              stopped: logicStopped ? 1 : 0,
              used: logicUsed ? 1 : 0,
            };
            sheetData.push(personData);
          }
        }
      }

      // ID順（1, 2, 3...）にソートしてデータを格納
      sheetData.sort((a, b) => a.id - b.id);

      let debugImageURL = null;
      if (generateDebug) {
        debugImageURL = createDebugImage(dst, true);
      }

      // 正常終了時のクリーンアップ
      if (srcOriginal) srcOriginal.delete();
      if (src) src.delete();
      if (dst) dst.delete();
      if (gray) gray.delete();
      if (thresholded) thresholded.delete();
      if (contours) contours.delete();
      if (hierarchy) hierarchy.delete();
      if (srcTri) srcTri.delete();
      if (dstTri) dstTri.delete();
      if (warpedGray) warpedGray.delete();
      if (warpedThresh) warpedThresh.delete();
      if (debugSrc) debugSrc.delete();
      markers.forEach(m => m.approx.delete()); // 残ったマーカーのメモリ解放

      return {
        success: true,
        data: { filename, peopleCount: sheetData.length, rows: sheetData },
        debug: generateDebug ? { image: debugImageURL, cells: debugCells, isError: false } : null
      };

    } catch (e) {
      console.error(e);
      let errorImgURL = null;
      if (generateDebug && debugSrc && !debugSrc.isDeleted()) {
          const canvas = document.createElement('canvas');
          cv.imshow(canvas, debugSrc);
          errorImgURL = canvas.toDataURL();
      }
      
      // エラー時の強制クリーンアップ（漏れ防止）
      try {
        if (srcOriginal && !srcOriginal.isDeleted()) srcOriginal.delete();
        if (src && !src.isDeleted()) src.delete();
        if (dst && !dst.isDeleted()) dst.delete();
        if (gray && !gray.isDeleted()) gray.delete();
        if (thresholded && !thresholded.isDeleted()) thresholded.delete();
        if (contours && !contours.isDeleted()) contours.delete();
        if (hierarchy && !hierarchy.isDeleted()) hierarchy.delete();
        if (srcTri && !srcTri.isDeleted()) srcTri.delete();
        if (dstTri && !dstTri.isDeleted()) dstTri.delete();
        if (warpedGray && !warpedGray.isDeleted()) warpedGray.delete();
        if (warpedThresh && !warpedThresh.isDeleted()) warpedThresh.delete();
        if (debugSrc && !debugSrc.isDeleted()) debugSrc.delete();
        markers.forEach(m => { if(!m.approx.isDeleted()) m.approx.delete(); });
      } catch (cleanupErr) {
          console.error("Cleanup error:", cleanupErr);
      }

      return { success: false, error: e.message, debug: { image: errorImgURL, isError: true } };
    }
  };

  /**
   * CSVエクスポート
   */
  const downloadCSV = (type) => {
    if (results.length === 0) return;

    let csvContent = "";
    
    if (type === 'regression') {
      csvContent += "Filename,ID,Male,Female,Noticed,Stopped,Used\n";
      results.forEach(fileResult => {
        fileResult.rows.forEach(row => {
           csvContent += `${fileResult.filename},${row.id},${row.logic.male},${row.logic.female},${row.logic.noticed},${row.logic.stopped},${row.logic.used}\n`;
        });
      });
    } else if (type === 'stats') {
      csvContent += "Filename,TotalPeople,Males,Females,Conv_Noticed,Conv_Stopped,Conv_Used\n";
      results.forEach(res => {
        const total = res.peopleCount;
        const males = res.rows.filter(r => r.logic.male).length;
        const females = res.rows.filter(r => r.logic.female).length;
        const noticed = res.rows.filter(r => r.logic.noticed).length;
        const stopped = res.rows.filter(r => r.logic.stopped).length;
        const used = res.rows.filter(r => r.logic.used).length;
        
        csvContent += `${res.filename},${total},${males},${females},${noticed},${stopped},${used}\n`;
      });
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `marksheet_${type}_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex justify-between items-start border-b border-slate-200 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-pink-500 rounded-2xl shadow-lg">
              <CheckCircle className="text-white w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-900 tracking-tight">MarkSheet Reader Pro</h1>
              <p className="text-slate-500">Aruco補正・階層分析対応</p>
            </div>
          </div>
          <button 
            onClick={() => setShowGuide(!showGuide)}
            className="flex items-center space-x-2 text-slate-500 hover:text-pink-500 transition-colors px-4 py-2 rounded-lg hover:bg-slate-100"
          >
            <HelpCircle className="w-5 h-5" />
            <span className="font-bold">使い方・理論ガイド</span>
            {showGuide ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </header>

        {/* Guide Section */}
        {showGuide && (
          <div className="bg-white rounded-2xl shadow-lg border border-pink-100 overflow-hidden animate-fade-in">
             <div className="p-6 space-y-8">
                {/* 1. Theory */}
                <section>
                   <h3 className="text-xl font-bold text-slate-800 flex items-center gap-2 mb-4">
                     <Activity className="w-6 h-6 text-pink-500" />
                     仕掛学：行動変容の階層モデル
                   </h3>
                   <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                     <p className="text-slate-600 mb-4 leading-relaxed">
                       本アプリは、通行人の行動を「関与の深さ」に応じて階層的に記録・分析するためのツールです。
                       <a href="https://www.shikakeology.org/pdf/SIG-TBC-012-03.pdf" target="_blank" rel="noreferrer" className="text-pink-500 hover:underline inline-flex items-center gap-1 mx-1">
                          仕掛学の研究<ExternalLink className="w-3 h-3"/>
                       </a>
                       に基づく以下のプロセスを前提としています。
                     </p>
                     
                     <div className="flex flex-col md:flex-row gap-4 items-center justify-center my-6">
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 text-center w-full md:w-32">
                           <div className="font-bold text-slate-400 mb-1 text-sm">Step 0</div>
                           <div className="font-bold text-slate-800">通行</div>
                           <div className="text-xs text-slate-500 mt-1">性別チェック</div>
                        </div>
                        <div className="text-slate-300">▶</div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 text-center w-full md:w-32">
                           <div className="font-bold text-blue-400 mb-1 text-sm">Step 1</div>
                           <div className="font-bold text-slate-800">気づいた</div>
                           <div className="text-xs text-slate-500 mt-1">Look</div>
                        </div>
                        <div className="text-slate-300">▶</div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 text-center w-full md:w-32">
                           <div className="font-bold text-blue-500 mb-1 text-sm">Step 2</div>
                           <div className="font-bold text-slate-800">立ち止まり</div>
                           <div className="text-xs text-slate-500 mt-1">Stop</div>
                        </div>
                        <div className="text-slate-300">▶</div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 text-center w-full md:w-32 ring-2 ring-pink-500 ring-offset-2">
                           <div className="font-bold text-pink-500 mb-1 text-sm">Step 3</div>
                           <div className="font-bold text-slate-800">使った</div>
                           <div className="text-xs text-slate-500 mt-1">Use</div>
                        </div>
                     </div>
                     <p className="text-xs text-slate-500 bg-white p-3 rounded border border-slate-200">
                        <strong>💡 自動補完ロジック:</strong> 上位の行動（例：使った）が記録されている場合、下位のプロセス（気づいた、立ち止まった）も「通過した」とみなして自動的にデータ上のフラグをONにします。
                     </p>
                   </div>
                </section>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* 2. Usage */}
                  <section>
                    <h3 className="text-lg font-bold text-slate-800 mb-3 flex items-center gap-2">
                       <Settings className="w-5 h-5 text-slate-400" />
                       使用方法
                    </h3>
                    <ol className="list-decimal list-inside space-y-2 text-sm text-slate-600 bg-slate-50 p-4 rounded-xl border border-slate-200">
                       <li>所定のマークシートを印刷し、観察データを記入します。</li>
                       <li>四隅のArUcoマーカーが入るように写真を撮影します。</li>
                       <li>本ツールのアップロードエリアに画像をドラッグ＆ドロップします。</li>
                       <li>解析ボタンを押し、デバッグ画像で行ズレがないか確認します。</li>
                       <li>必要に応じて「黒色判定しきい値」を調整し、再解析します。</li>
                       <li>CSVをダウンロードして分析に使用します。</li>
                    </ol>
                  </section>
                  
                  {/* 3. Download */}
                  <section>
                    <h3 className="text-lg font-bold text-slate-800 mb-3 flex items-center gap-2">
                       <Download className="w-5 h-5 text-slate-400" />
                       リソース
                    </h3>
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3">
                       <p className="text-sm text-slate-600">専用マークシートはこちらからダウンロードしてください。</p>
                       <a 
                         href="./assets/実験シート240v2.pdf" 
                         onClick={(e) => e.preventDefault()}
                         className="flex items-center justify-between p-3 bg-white border border-slate-200 rounded-lg hover:border-pink-300 hover:text-pink-600 transition-colors group"
                       >
                          <span className="font-bold text-sm">📄 実験記録シート (Ver 2.0).pdf</span>
                          <Download className="w-4 h-4 text-slate-300 group-hover:text-pink-500" />
                       </a>
                       <p className="text-xs text-slate-400 text-right">※ 現在準備中です（ダミーリンク）</p>
                    </div>
                  </section>
                </div>
             </div>
             <div className="bg-slate-50 p-3 text-center border-t border-slate-100">
                <button onClick={() => setShowGuide(false)} className="text-sm text-pink-500 font-bold hover:underline">
                   ガイドを閉じる
                </button>
             </div>
          </div>
        )}

        {/* Status Check */}
        {!cvLoaded && (
          <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-xl flex items-center space-x-3 animate-pulse">
            <Activity className="text-yellow-600 w-5 h-5" />
            <span className="text-yellow-800 font-medium">OpenCVエンジンを読み込んでいます...</span>
          </div>
        )}

        {/* Upload Area */}
        <div 
          className={`transition-all duration-300 ${processing ? 'opacity-50 pointer-events-none' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <label className={`block w-full cursor-pointer group ${isDragging ? 'scale-[1.02]' : ''} transition-transform duration-200`}>
            <input 
              type="file" 
              multiple 
              accept="image/*" 
              onChange={handleImageUpload} 
              className="hidden" 
              disabled={!cvLoaded}
            />
            <div className={`bg-white border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center transition-colors ${
              isDragging 
                ? 'border-pink-500 bg-pink-50' 
                : 'border-slate-300 group-hover:border-pink-400 group-hover:bg-pink-50'
            }`}>
              <Upload className={`w-12 h-12 mb-4 transition-colors ${
                isDragging ? 'text-pink-600' : 'text-slate-400 group-hover:text-pink-500'
              }`} />
              <p className={`text-lg font-medium transition-colors ${
                isDragging ? 'text-pink-700' : 'text-slate-600 group-hover:text-pink-600'
              }`}>
                {isDragging ? 'ここにドロップ！' : 'ここにマークシート画像をドラッグ＆ドロップ'}
              </p>
              <p className="text-sm text-slate-400 mt-2">
                複数ファイル対応。ファイル名順に処理されます (Arucoマーカー必須)
              </p>
            </div>
          </label>
        </div>

        {/* Setting & Control Area */}
        {images.length > 0 && !processing && (
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 animate-fade-in space-y-4">
            <div className="flex items-center gap-2 text-slate-700 font-bold border-b border-slate-100 pb-2">
               <Settings className="w-5 h-5" />
               <span>解析設定</span>
            </div>
            
            <div className="flex flex-col md:flex-row md:items-center gap-6">
               <div className="flex-1">
                  <div className="flex justify-between mb-2">
                    <label className="text-sm font-medium text-slate-600">黒色判定しきい値 (Threshold)</label>
                    <span className="font-mono font-bold text-pink-600 bg-pink-50 px-2 rounded">{bwThreshold}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.05" 
                    max="0.5" 
                    step="0.01" 
                    value={bwThreshold}
                    onChange={(e) => setBwThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-pink-500"
                  />
                  <p className="text-xs text-slate-400 mt-1">
                     ※ 値を下げると薄いマークも検知しますが、ゴミを拾いやすくなります。<br/>
                     ※ デフォルトは 0.2 です。
                  </p>
               </div>
               
               <div className="flex items-end">
                 <button
                   onClick={processImages}
                   className="w-full md:w-auto bg-pink-500 text-white px-8 py-3 rounded-full font-bold shadow-lg hover:bg-pink-600 transition-all hover:scale-105 active:scale-95 flex items-center justify-center gap-2 whitespace-nowrap"
                 >
                   {results.length > 0 ? <RefreshCw className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                   {results.length > 0 ? "設定を適用して再解析" : `解析を開始する (${images.length}枚)`}
                 </button>
               </div>
            </div>
          </div>
        )}

        {/* Processing Indicator */}
        {processing && (
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold text-slate-700">解析中...</span>
              <span className="text-pink-600 font-mono">{progress.current} / {progress.total}</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div 
                className="bg-pink-500 h-full transition-all duration-300 ease-out"
                style={{ width: `${(progress.current / progress.total) * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Results Area */}
        {results.length > 0 && !processing && (
          <div className="space-y-6 animate-fade-in">
            {/* Action Buttons */}
            <div className="flex gap-4">
              <button 
                onClick={() => downloadCSV('regression')}
                className="flex-1 bg-slate-900 text-white py-4 rounded-xl font-bold shadow-lg hover:bg-slate-800 flex items-center justify-center gap-2 transition-transform active:scale-95"
              >
                <Database className="w-5 h-5" />
                回帰分析用CSV (Raw Data)
              </button>
              <button 
                onClick={() => downloadCSV('stats')}
                className="flex-1 bg-white border-2 border-slate-200 text-slate-700 py-4 rounded-xl font-bold hover:bg-slate-50 flex items-center justify-center gap-2 transition-transform active:scale-95"
              >
                <FileText className="w-5 h-5" />
                基本統計CSV (Summary)
              </button>
            </div>

            {/* Results Table */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                <h3 className="font-bold text-slate-700">解析結果サマリ</h3>
                <span className="text-xs text-slate-400 bg-white px-2 py-1 rounded border">Total Files: {results.length}</span>
              </div>
              <div className="max-h-60 overflow-y-auto">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-slate-500 uppercase bg-slate-50 sticky top-0">
                    <tr>
                      <th className="px-6 py-3">File</th>
                      <th className="px-6 py-3">Count (People)</th>
                      <th className="px-6 py-3">Valid Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((res, idx) => (
                      <tr key={idx} className="border-b last:border-0 hover:bg-slate-50">
                        <td className="px-6 py-4 font-medium text-slate-900">{res.filename}</td>
                        <td className="px-6 py-4">
                          <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-bold">
                            {res.peopleCount}人
                          </span>
                        </td>
                        <td className="px-6 py-4 text-slate-500">
                          {Math.round((res.peopleCount / 240) * 100)}% Capacity
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Debug View (Always show if available, even on error) */}
        {debugData && (
          <div className={`rounded-2xl shadow-sm border p-6 space-y-4 ${debugData.isError ? 'bg-red-50 border-red-200' : 'bg-white border-slate-200'}`}>
            <div className="flex items-center gap-2 mb-2">
              <Eye className={`w-5 h-5 ${debugData.isError ? 'text-red-500' : 'text-pink-500'}`} />
              <h3 className={`font-bold ${debugData.isError ? 'text-red-700' : 'text-slate-700'}`}>
                {debugData.isError ? 'エラーデバッグ表示（二値化・輪郭検出）' : 'デバッグ表示（歪み補正・読み取り）'}
              </h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-xs font-bold uppercase text-slate-400 mb-2">
                  {debugData.isError ? 'Threshold & Contours (Green=Quad, Red=Corner)' : 'Warped Image (with Grid)'}
                </h4>
                {debugData.image ? (
                  <div className="relative border border-slate-300 rounded overflow-hidden max-h-96 w-full bg-slate-100">
                    <img src={debugData.image} alt="Debug" className="w-full h-full object-contain" />
                  </div>
                ) : (
                  <div className="h-40 flex items-center justify-center bg-slate-100 text-slate-400 text-xs">No Image Available</div>
                )}
                <p className="text-xs text-slate-400 mt-2">
                  {debugData.isError 
                    ? '※ エラー発生時の画像処理状況です。緑の枠が検出された四角形、赤点が採用されたコーナーです。真っ黒/真っ白になっていないか確認してください。'
                    : '※ 補正画像に水色のグリッド線を表示しています。マーク枠とズレていないか確認してください。'}
                </p>
              </div>
              
              {!debugData.isError && results.length > 0 && (
                <div>
                  <h4 className="text-xs font-bold uppercase text-slate-400 mb-2">First Row Detection (Cell Images)</h4>
                  <div className="bg-slate-900 p-4 rounded-xl overflow-y-auto max-h-96">
                    {/* Iterate over 4 potential people in the row */}
                    {[0, 1, 2, 3].map(personIdx => {
                      // ID calculation for display: Block Offset (60) * BlockIdx + Row (1)
                      const displayId = (personIdx * 60) + 1;
                      return (
                        <div key={personIdx} className="mb-4 border-b border-slate-700 pb-2 last:border-0">
                          <div className="text-pink-400 font-bold text-xs mb-2">
                             Block {personIdx + 1} (Start ID: {displayId})
                          </div>
                          <div className="grid grid-cols-6 gap-2">
                             {/* Headers */}
                             {DATA_LABELS.map(l => <div key={l} className="text-[10px] text-slate-500 text-center">{l.slice(0,4)}</div>)}
                             
                             {/* Cells (Looking for cells that match this specific ID) */}
                             {debugData.cells && debugData.cells
                               .filter(cell => cell.person === displayId)
                               .map((cell, cIdx) => (
                                 <div key={cIdx} className="flex flex-col items-center space-y-1">
                                    <div className={`relative p-0.5 rounded ${cell.checked ? 'bg-green-500' : 'bg-slate-700'}`}>
                                      <img src={cell.image} alt="cell" className="w-full h-auto rounded-sm min-h-[30px] bg-white" />
                                    </div>
                                    <span className={`text-[10px] font-mono ${cell.checked ? 'text-green-400' : 'text-slate-500'}`}>
                                      {cell.ratio}
                                    </span>
                                 </div>
                               ))
                             }
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Logs */}
        <div className="bg-slate-900 rounded-2xl p-6 text-xs font-mono max-h-40 overflow-y-auto shadow-inner">
          <div className="flex items-center gap-2 mb-2 sticky top-0 bg-slate-900 pb-2 border-b border-slate-800">
             <AlertCircle className="w-4 h-4 text-slate-400" />
             <span className="text-slate-400 font-bold">System Logs</span>
          </div>
          {logs.map((log, i) => (
            <div key={i} className={`mb-1 ${log.type === 'error' ? 'text-red-400' : 'text-slate-300'}`}>
              <span className="text-slate-500 mr-2">[{log.time}]</span>
              {log.msg}
            </div>
          ))}
          {logs.length === 0 && <span className="text-slate-600">待機中...</span>}
        </div>

      </div>
    </div>
  );
};

export default App;
