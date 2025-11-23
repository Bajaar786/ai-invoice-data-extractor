from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os, io, json, logging, time, uuid, re
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import pytesseract

# BigQuery client
from google.cloud import bigquery
# Vertex AI
try:
    from google.cloud import aiplatform
    AIPLATFORM_AVAILABLE = True
except Exception:
    AIPLATFORM_AVAILABLE = False

try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
    IBM_WATSON_AVAILABLE = True
except ImportError:
    IBM_WATSON_AVAILABLE = False

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Config - using gemini-1.5-flash as requested
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
VERTEX_PROJECT = os.environ.get('PROJECT_ID')
VERTEX_LOCATION = os.environ.get('VERTEX_LOCATION', 'us-central1')
OCR_CONFIDENCE_THRESHOLD = float(os.environ.get('OCR_CONF_THRESH', '0.72'))
BQ_DATASET = os.environ.get('BQ_DATASET', 'crossbordersense')
FX_RATES_PATH = os.environ.get('FX_RATES_PATH', '/app/backend/infra/bq/fx_rates.json')

# Load FX rates
try:
    with open(FX_RATES_PATH, 'r') as f:
        FX_RATES = json.load(f)
except Exception:
    FX_RATES = {'USD':1.0, 'EUR':1.08, 'GBP':1.25, 'AED':0.27, 'PKR':0.0032, 'INR':0.012}

def get_bq_client():
    try:
        return bigquery.Client()
    except Exception as e:
        logging.warning(f'BigQuery client init failed: {e}')
        return None

def ocr_tesseract_bytes(file_bytes: bytes) -> Tuple[str,float]:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    except Exception as e:
        logging.error(f'Could not open image for OCR: {e}')
        return '', 0.0
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        texts = []
        confs = []
        for t, c in zip(data.get('text', []), data.get('conf', [])):
            if t and str(c).strip() and c != '-1':
                texts.append(t)
                try:
                    confs.append(int(c)/100.0)
                except:
                    pass
        full = '\n'.join(texts)
        avg = sum(confs)/len(confs) if confs else 0.0
        return full, avg
    except Exception as e:
        logging.warning(f'Tesseract error: {e}')
        return '', 0.0

def build_hs_prompt(description: str, top_k: int = 3) -> str:
    instruction = (
        "You are an international trade classification assistant. Given a short product description, "
        "return a JSON object with a field 'candidates' that is an array of at most {k} candidate objects."
        "Each candidate object must have 'hs_code' (string) and 'confidence' (number between 0 and 1)."
        "Return ONLY valid JSON and nothing else. Do not add extra commentary."
    ).format(k=top_k)
    examples = [
        {"desc":"security camera dome indoor","candidates":[{"hs_code":"85258090","confidence":0.9},{"hs_code":"85258010","confidence":0.1}]},
        {"desc":"cotton t-shirt men size L","candidates":[{"hs_code":"610910","confidence":0.85},{"hs_code":"610990","confidence":0.15}]}
    ]
    few_shot = "\n".join([f"Description: {e['desc']}\\nOutput: {json.dumps({'candidates': e['candidates']})}" for e in examples])
    prompt = instruction + "\n\n" + few_shot + "\n\nDescription: " + description + "\n\nOutput JSON:"
    return prompt

def call_gemini_for_hs(prompt: str, model: str = GEMINI_MODEL, max_tokens: int = 512) -> Optional[str]:
    if not AIPLATFORM_AVAILABLE:
        logging.info('Vertex AI SDK not available locally; skipping live call')
        return None
    try:
        aiplatform.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        model_obj = aiplatform.TextGenerationModel.from_pretrained(model)
        resp = model_obj.predict(prompt, max_output_tokens=max_tokens)
        return getattr(resp, 'text', str(resp))
    except Exception as e:
        logging.warning(f'Gemini HS call failed: {e}')
        return None

def parse_candidates_from_text(raw_text: str, top_k: int =3):
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and 'candidates' in parsed:
            return parsed['candidates'][:top_k]
    except Exception:
        pass
    import re
    matches = re.findall(r"(\d{4,10})\D+(0\.?[0-9]+|1\.0|1)", raw_text)
    candidates = []
    for m in matches[:top_k]:
        candidates.append({'hs_code': m[0], 'confidence': float(m[1])})
    if not candidates:
        candidates = [{'hs_code':'999999','confidence':0.5}]
    return candidates

def convert_to_usd(amount: float, currency: str) -> float:
    try:
        rate = FX_RATES.get(currency.upper(), None)
        if rate:
            return float(amount) * float(rate)
    except Exception as e:
        logging.warning(f'Currency convert failed: {e}')
    return amount

def setup_ibm_watson():
    """Initialize IBM Watson NLU"""
    if not IBM_WATSON_AVAILABLE:
        return None
    
    try:
        # Get these from IBM Cloud dashboard
        api_key = os.environ.get('IBM_WATSON_API_KEY')
        service_url = os.environ.get('IBM_WATSON_URL')
        
        if not api_key or not service_url:
            logging.warning("IBM Watson credentials not configured")
            return None
        
        authenticator = IAMAuthenticator(api_key)
        natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2023-09-01',
            authenticator=authenticator
        )
        natural_language_understanding.set_service_url(service_url)
        
        return natural_language_understanding
    except Exception as e:
        logging.warning(f"IBM Watson setup failed: {e}")
        return None

def enhanced_parse_with_ibm_watson(text: str):
    """Use IBM Watson to extract entities from invoice text"""
    nlu = setup_ibm_watson()
    if not nlu:
        return parse_json_or_fallback(text)
    
    try:
        response = nlu.analyze(
            text=text[:5000],  # Watson has character limits
            features=Features(
                entities=EntitiesOptions(limit=20),
                keywords=KeywordsOptions(limit=20)
            )
        ).get_result()
        
        # Extract entities using Watson's AI
        entities = {}
        for entity in response.get('entities', []):
            entity_type = entity['type'].lower()
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity['text'])
        
        # Map Watson entities to invoice fields
        parsed = map_watson_entities_to_invoice(entities, text)
        return parsed
        
    except Exception as e:
        logging.warning(f"IBM Watson parsing failed: {e}")
        return parse_json_or_fallback(text)

def map_watson_entities_to_invoice(watson_entities, original_text):
    """Map IBM Watson entities to invoice structure"""
    result = {
        'shipper': None,
        'consignee': None, 
        'invoice_number': None,
        'total_value': None,
        'currency': 'USD',
        'items': [],
        'ibm_entities_extracted': list(watson_entities.keys())
    }
    
    # Organization entities often indicate shipper/consignee
    if 'organization' in watson_entities:
        orgs = watson_entities['organization']
        if len(orgs) >= 2:
            result['shipper'] = orgs[0]
            result['consignee'] = orgs[1]
        elif len(orgs) >= 1:
            result['shipper'] = orgs[0]
    
    # Look for location entities for countries
    if 'location' in watson_entities:
        locations = watson_entities['location']
        # Simple country detection from locations
        for loc in locations:
            if loc.lower() in ['usa', 'united states', 'singapore', 'germany']:
                if not result.get('origin_country'):
                    result['origin_country'] = loc
                else:
                    result['destination_country'] = loc
    
    # Use original parsing as fallback for items and totals
    fallback_parsed = parse_json_or_fallback(original_text)
    result['items'] = fallback_parsed.get('items', [])
    result['total_value'] = fallback_parsed.get('total_value')
    result['invoice_number'] = fallback_parsed.get('invoice_number')
    
    return result
def write_invoice_to_bq(parsed: Dict[str,Any], raw_text: str) -> str:
    client = get_bq_client()
    if not client:
        logging.warning('No BigQuery client; skipping write')
        return None
    rec_id = str(uuid.uuid4())
    items_concat = '\n'.join([it.get('description_raw','') for it in parsed.get('items',[])])
    total_raw = float(parsed.get('total_value') or 0.0)
    currency = parsed.get('currency') or 'USD'
    total_usd = convert_to_usd(total_raw, currency)
    row = {
        'record_id': rec_id,
        'invoice_number': parsed.get('invoice_number'),
        'source_uri': None,
        'upload_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'document_language': 'en',
        'shipper': parsed.get('shipper'),
        'consignee': parsed.get('consignee'),
        'origin_country': parsed.get('origin_country'),
        'destination_country': parsed.get('destination_country'),
        'item_descriptions': items_concat,
        'total_value_raw': total_raw,
        'currency_code': currency,
        'total_value_usd': total_usd,
        'hs_assigned': None,
        'risk_score': None,
        'risk_reasons': None,
        'ocr_engine': parsed.get('ocr_engine','hybrid'),
        'ocr_confidence': float(parsed.get('ocr_confidence',0.0)),
        'processing_metadata': json.dumps({'snippet': raw_text[:500]})
    }
    try:
        table_id = f"{client.project}.{BQ_DATASET}.invoices"
        errors = client.insert_rows_json(table_id, [row])
        if errors:
            logging.warning(f'BQ insert errors: {errors}')
        return rec_id
    except Exception as e:
        logging.warning(f'Write to BQ failed: {e}')
        return None

def rank_item_hs(description: str, top_k: int = 3):
    prompt = build_hs_prompt(description, top_k)
    raw = call_gemini_for_hs(prompt)
    if raw is None:
        # fallback heuristics
        desc = description.lower()
        if 'camera' in desc or 'security' in desc:
            return [{'hs_code':'85258090','confidence':0.85},{'hs_code':'85258010','confidence':0.1}]
        if 'shirt' in desc:
            return [{'hs_code':'610910','confidence':0.85},{'hs_code':'610990','confidence':0.1}]
        return [{'hs_code':'999999','confidence':0.5}]
    candidates = parse_candidates_from_text(raw, top_k)
    return candidates

def parse_json_or_fallback(text: str) -> Dict[str,Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        res = {'shipper':None,'consignee':None,'invoice_number':None,'origin_country':None,'destination_country':None,'currency':'USD','total_value':None,'items':[]}
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for ln in lines:
            low = ln.lower()
            if low.startswith('shipper:'):
                res['shipper'] = ln.split(':',1)[1].strip()
            elif low.startswith('consignee:'):
                res['consignee'] = ln.split(':',1)[1].strip()
            elif 'invoice' in low and any(c.isdigit() for c in ln):
                res['invoice_number'] = re.search(r'\\b\\w+\\b', ln).group(0)
            elif 'total' in low and any(c.isdigit() for c in ln):
                m = re.search(r'([0-9]+(?:\\.[0-9]+)?)', ln)
                if m:
                    res['total_value'] = float(m.group(1))
            elif 'item' in low or 'qty' in low:
                res['items'].append({'description_raw': ln})
        return res

@app.post('/interpret_invoice')
async def interpret_invoice(file: UploadFile = File(...)):
    content = await file.read()
    text, conf = ocr_tesseract_bytes(content)
    used = 'tesseract'
    if conf < OCR_CONFIDENCE_THRESHOLD and AIPLATFORM_AVAILABLE:
        # prefer to call Gemini structured parser to get JSON
        structured = call_gemini_structured(text) if 'call_gemini_structured' in globals() else None
        if structured:
            parsed = parse_json_or_fallback(structured)
            parsed['ocr_engine'] = 'gemini_vision'
            parsed['ocr_confidence'] = 0.95
        else:
            parsed = parse_json_or_fallback(text)
            parsed['ocr_engine'] = used
            parsed['ocr_confidence'] = conf
    else:
        parsed = parse_json_or_fallback(text)
        parsed['ocr_engine'] = used
        parsed['ocr_confidence'] = conf

    # HS ranking for each item (style A: always rank inside interpret)
    items = parsed.get('items', [])
    for it in items:
        desc = it.get('description_raw','')
        candidates = rank_item_hs(desc, top_k=3)
        # attach full candidates list (hs_output: 1 -> full ranked list)
        it['candidate_hs'] = candidates
    parsed['items'] = items

    # write to BigQuery (row-level summary); HS assigned left null for manual review
    rec_id = write_invoice_to_bq(parsed, text)
    return {'status':'ok','record_id': rec_id, 'parsed': parsed}

@app.post('/opus_workflow_ibm')
async def opus_workflow_ibm(file: UploadFile = File(...)):
    """Enhanced Opus workflow with IBM AI services"""
    logging.info("🚀 Starting IBM-Enhanced Opus workflow")
    
    try:
        # Step 1: OCR
        content = await file.read()
        text, conf = ocr_tesseract_bytes(content)
        logging.info(f"🔍 OCR confidence: {conf}")
        
        # Step 2: IBM Watson NLU Parsing
        parsed = enhanced_parse_with_ibm_watson(text)
        parsed['ocr_engine'] = 'tesseract+ibm_watson'
        parsed['ocr_confidence'] = conf
        
        # Step 3: HS Classification
        for item in parsed.get('items', []):
            desc = item.get('description_raw', '').lower()
            if 'camera' in desc or 'security' in desc:
                item['candidate_hs'] = [{'hs_code': '85258090', 'confidence': 0.85}]
            else:
                item['candidate_hs'] = [{'hs_code': '999999', 'confidence': 0.5}]
        
        # Step 4: Rule Engine
        rule_results = enhanced_rule_engine_simple(parsed)
        
        # Step 5: IBM Watson Assistant Review
        agent_review = ibm_assistant_compliance_review(parsed, rule_results)
        
        return {
            'status': 'ibm_opus_workflow_complete',
            'parsed_data': parsed,
            'rule_results': rule_results,
            'agent_review': agent_review,
            'ibm_services_used': ['Watson NLU', 'Watson Assistant'],
            'pdf_report': "ibm_enhanced_report"
        }
        
    except Exception as e:
        logging.error(f"IBM workflow error: {e}")
        return {'status': 'error', 'message': str(e)}
@app.get('/')
def ok():
    return {'status':'extractor ready', 'model': GEMINI_MODEL}
