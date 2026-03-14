import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
import io

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def get_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    content_type = response.headers.get('Content-Type', '')
    
    # PDF인 경우
    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
        if pypdf is None:
            raise ImportError("pypdf 패키지가 설치되지 않았습니다. 'pip install pypdf' 명령어로 설치해주세요.")
        
        pdf_file = io.BytesIO(response.content)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    else:
        # 일반 웹페이지 (HTML) 인 경우
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # PubMed인 경우 초록 및 본문 추출 로직 추가
        if 'pubmed.ncbi.nlm.nih.gov' in url:
            pubmed_text = ""
            
            # 초록(Abstract) 추출 시도
            abstract_div = soup.find('div', id='enc-abstract') or soup.find('div', class_='abstract-content selected')
            if abstract_div:
                pubmed_text += "--- PubMed Abstract ---\n"
                pubmed_text += abstract_div.get_text(separator='\n', strip=True) + "\n\n"
            
            # 본문(Body/Full text) 추출 시도 (PubMed Central 등에서 무료로 제공되는 경우)
            body_div = soup.find('div', id='enc-body')
            if body_div:
                pubmed_text += "--- PubMed Full Text Option ---\n"
                pubmed_text += body_div.get_text(separator='\n', strip=True) + "\n\n"
                
            if pubmed_text.strip():
                return pubmed_text
                
        # PubMed가 아니거나 (또는 추출에 실패한 경우) 기본 로직 진행
        # arXiv나 일반 사이트의 초록/본문을 가져오기 위해 주로 <p> 태그 추출
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # 만약 본문이 너무 짧다면 전체 텍스트 추출 시도
        if len(text) < 500:
            text = soup.get_text(separator='\n', strip=True)
            
        return text

def analyze_paper(text, api_key=None):
    if OpenAI is None:
        raise ImportError("openai 패키지가 설치되지 않았습니다. 'pip install openai' 명령어로 설치해주세요.")
    
    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)
    
    # 컨텍스트 길이(Token 제한)를 고려하여 앞부분 텍스트만 사용 (약 15000단어/40000자 내외)
    # 논문의 Research Objective와 Dataset은 보통 초록(Abstract)이나 Introduction, Method 앞부분 제안에 등장하므로 충분합니다.
    max_chars = 40000 
    truncated_text = text[:max_chars]
    
    prompt = f"""
다음 제공된 논문 텍스트를 읽고, 아래 두 가지 항목을 매우 명확하고 구체적으로 추출해주세요:
1. 연구 목적 (Research Objective / Motivation)
2. 사용된 데이터셋 (Datasets Used) - 만약 별도의 데이터셋이 명시되어 있지 않다면 '명시되지 않음' 또는 추론된 내용을 적어주세요.

글머리 기호를 사용하여 가독성 있게 한국어로 정리해주세요.

텍스트:
{truncated_text}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # 비용 효율적인 최신 모델
        messages=[
            {"role": "system", "content": "당신은 세계 최고의 논문 분석 AI 연구원입니다. 주어진 내용 안에서만 근거를 찾아 답변하고 없는 내용을 지어내지 마세요."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="논문 링크에서 '연구 목적'과 '데이터셋'을 추출해주는 AI 스크립트입니다.")
    parser.add_argument("url", nargs='?', help="분석할 논문의 URL (PDF 직접 링크 또는 arXiv 등 웹페이지 링크)")
    args = parser.parse_args()
    
    url = args.url
    if not url:
        print("\n💡 [안내] 실행 시 URL이 입력되지 않았습니다.")
        url = input("🔗 분석할 논문의 URL을 직접 입력하세요:\n👉 (붙여넣기 후 엔터) ").strip()
        if not url:
            print("❌ 에러: URL이 입력되지 않았습니다. 프로그램을 종료합니다.")
            sys.exit(1)
            
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 에러: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("터미널에 다음 명령어를 입력하여 API 키를 설정해주세요:")
        print("export OPENAI_API_KEY='sk-여러분의_실제_API_키'")
        sys.exit(1)
        
    print(f"⏳ [{url}]에서 텍스트를 다운로드 및 추출하는 중...")
    try:
        text = get_text_from_url(url)
        if not text.strip():
            print("❌ 텍스트를 추출하지 못했습니다. 링크가 올바른지, 접근 권한이 필요한 페이지인지 확인해주세요.")
            sys.exit(1)
            
        print(f"✅ 텍스트 추출 완료 (길이: {len(text):,}자). AI 모델로 분석을 시작합니다...")
        
        result = analyze_paper(text, api_key)
        
        print("\n" + "="*55)
        print("📊 [논문 핵심 요약 분석 결과]")
        print("="*55)
        print(result)
        print("="*55)
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ 웹 접속 오류 (HTTP Error): {e}")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
