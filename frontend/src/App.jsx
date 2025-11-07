import React, {useState} from 'react'
import './style.css'

export default function App(){
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const BACKEND_URL = "https://extractor-REPLACE_WITH_YOUR_URL.a.run.app";

  const upload = async ()=>{
    if(!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append('file', file);
    try{
      const res = await fetch(BACKEND_URL + '/interpret_invoice', { method: 'POST', body: fd });
      const j = await res.json();
      setResult(j);
    }catch(e){
      alert('Upload failed: ' + e);
    }finally{
      setLoading(false);
    }
  }

  return (<div className="container">
    <h1>CrossBorderSense — Demo</h1>
    <input type="file" onChange={e=>setFile(e.target.files[0])} />
    <button onClick={upload} disabled={loading}>{loading ? 'Processing...' : 'Upload & Analyze'}</button>
    {result && <div className="results">
      <h2>Parsed Items</h2>
      <pre style={{whiteSpace:'pre-wrap'}}>{JSON.stringify(result, null, 2)}</pre>
    </div>}
  </div>)
}
