import docx
import copy
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_border(cell, top='single', top_sz='4', bottom='single', bottom_sz='4'):
    tcPr = cell._tc.get_or_add_tcPr()
    xml = f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="{top}" w:sz="{top_sz}" w:space="0" w:color="auto"/>
            <w:left w:val="none"/>
            <w:bottom w:val="{bottom}" w:sz="{bottom_sz}" w:space="0" w:color="auto"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    '''
    tcPr.append(parse_xml(xml))

def set_cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_run_font(run, font_size_pt=10, bold=False, italic=False, font_name="바탕체", ascii_font="Times New Roman"):
    run.font.name = ascii_font
    run.font.size = Pt(font_size_pt)
    run.bold = bold
    run.italic = italic
    rPr = run._r.get_or_add_rPr()
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:eastAsia="{font_name}" w:ascii="{ascii_font}" w:hAnsi="{ascii_font}"/>')
    rPr.append(rFonts)

def build_humantech_korean_doc_perfect(template_path, output_path):
    doc = docx.Document(template_path)
    body = doc._body._element

    # Consolidate section 0 sectPr to replace final sectPr
    p47 = body[47]
    sectPr0 = p47.xpath('./w:pPr/w:sectPr')[0]
    sectPr_final = body[-1]
    body.replace(sectPr_final, copy.deepcopy(sectPr0))
    p47.pPr.remove(sectPr0)

    # Fix header text in section
    s0 = doc.sections[0]
    for hp in s0.header.paragraphs:
        if hp.text.strip():
            hp.text = "33rd Humantech Paper Awards"
            if hp.runs:
                set_run_font(hp.runs[0], font_size_pt=9, bold=False)
            hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # Remove all children from body except first paragraph
    for child in list(body)[:-1]:
        body.remove(child)

    def add_p(text="", style="Normal", font_size=10, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=1, line_spacing=1.0):
        p = doc.add_paragraph(style=style)
        p.alignment = align
        pPr = p._p.get_or_add_pPr()
        sp_xml = f'<w:spacing {nsdecls("w")} w:before="{int(space_before*20)}" w:after="{int(space_after*20)}" w:line="{int(line_spacing*240)}" w:lineRule="auto"/>'
        pPr.append(parse_xml(sp_xml))
        if text:
            r = p.add_run(text)
            set_run_font(r, font_size_pt=font_size, bold=bold, italic=italic)
        return p

    # 1. Title (Abstract style, 20pt Bold, align left)
    add_p("IoT 엣지 환경을 위한 초저비용 상태 기반 이진화 신경망 보안 프로세서 (BitGuard-BNN)",
          style="Abstract", font_size=20, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=3, line_spacing=1.05)

    # 2. Abstract (Normal style, 10pt Bold, Justify)
    add_p("(Abstract) 사물인터넷(IoT) 침입 탐지 시스템(IDS) 연구는 단순 분류 모델의 정확도에만 치중되어 실제 하드웨어 제약, 전처리 병목, 단일 패킷(Stateless) 판단의 한계, 능동 방어 연계 부재를 해결하지 못하고 있다. 본 연구에서는 단순 AI 모델을 넘어, 실시간 스트리밍 특징 추출부터 1-bit BNN 가속, 오픈셋 제로데이 격리, 기기당 약 3 bytes 초경량 상태 추적, 단계적 방어 추천, 비트 패킹 엣지 배포까지 전 주기를 통합 설계한 초저비용 온디바이스 보안 프로세서 BitGuard-BNN을 제안한다. BitGuard는 (1) 페이로드 검사 없이 24차원 통계를 실시간 갱신하는 유계(bounded) 스트리밍 엔진, (2) 센싱 비용을 학습 목적함수에 반영하는 비용 인지 게이티드 BNN, (3) 정상 트래픽을 0-연산으로 통과시키는 Boolean-Tiny-Main 3단계 리스크 적응형 캐스케이드, (4) 미지의 공격을 격리하는 이중 조건 오픈셋 탐지기, (5) 5개 4-bit 카운터(기기당 약 3 bytes)로 저속 스텔스 공격을 포착하는 시간적 상태 머신, (6) BatchNorm 폴딩 및 종단간 패리티 검증 기반 1-bit 비트 패킹 배포 파이프라인으로 구성된다. 엄격한 무누출(Zero-leakage) 벤치마크 평가 결과, 제안 시스템은 FP32 대비 파라미터 메모리를 95.9% 절감(2.8 KB)하고 정상 패킷의 88.3%를 조기 종료하여 연산량을 96.4% 감축하면서도 99.64%의 Macro-F1과 99.98%의 고위험 재현율을 달성하였으며, 시간당 오경보율(FAR)을 92.4% 억제함을 실증하였다.",
          style="Normal", font_size=10, bold=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=1, space_after=3, line_spacing=1.0)

    # 3. 1. 서론 (11pt Bold)
    add_p("1. 서론", style="Normal", font_size=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=3, space_after=1)
    add_p("스마트 홈, 스마트 팩토리 등 사물인터넷(IoT) 기기의 폭발적 보급과 함께, 취약한 엣지 기기를 숙주로 삼아 대규모 분산 서비스 거부(DDoS) 공격이나 악성 스캔을 감행하는 IoT 봇넷(Mirai, Gafgyt, Reaper 등) 위협이 나날이 지능화되고 있다 [1]. 이에 대응하기 위해 딥러닝 기반 침입 탐지 시스템(IDS)이 활발히 연구되고 있으나, 수십 KB 수준의 가용 SRAM과 수백 mW 이하의 전력 예산을 갖는 초저전력 엣지 마이크로컨트롤러(MCU)에 직접 탑재하기에는 다음과 같은 치명적 한계를 지닌다 [2]:",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=1, line_spacing=1.0)
    add_p("첫째, 고비용 부동소수점(FP32) 연산과 수십~수백 KB의 가중치 메모리 요구량으로 인해 FPU가 부재한 저가형 MCU에서 인라인(In-line) 패킷 처리가 불가능하다. 둘째, 100개 이상의 복잡한 흐름 통계 계산 및 페이로드 심층 검사(DPI)로 인해 추론보다 전처리 단계에서 더 큰 센싱 에너지와 지연시간이 소모되며, TLS 암호화 트래픽 앞에서는 무력화된다. 셋째, 패킷 간 시간적 상관관계를 배제한 단일 패킷(Stateless) 판단 구조로 인해 일시적 버스트에 따른 오경보(False Alarm)를 남발하고, 패킷 전송 간격이 긴 저속 스텔스 스캔이나 C&C 비콘을 탐지하지 못한다. 넷째, 대다수 연구가 '공격 여부 분류'에만 그쳐 오탐 시 정상 기기의 통신을 강제 차단하는 사고를 방지할 안전한 단계적 능동 방어 체계가 부재하다. 다섯째, 이종 데이터셋 간 억지 제로 패딩 및 테스트 데이터 누수(Data Leakage)로 벤치마크 신뢰성이 결여되어 있다 [3].",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=1, line_spacing=1.0)
    add_p("본 연구에서는 이러한 문제를 근본적으로 해결하기 위해, '단순 AI 분류 모델'이 아닌 데이터 수집부터 특징 추출, BNN 추론, 상태 추적, 대응 판단, 엣지 배포까지 전 주기를 통합한 '초저비용 상태 기반 온디바이스 보안 프로세서' BitGuard-BNN을 제안한다.",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=2, line_spacing=1.0)

    # 4. 2. 제안 시스템 아키텍처 및 연구 방법론 (11pt Bold)
    add_p("2. 제안 시스템 아키텍처 및 연구 방법론", style="Normal", font_size=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=3, space_after=1)

    # 2.1
    add_p("2.1. 스트리밍 특징 프로세서 (MicroSecurityFeatureProcessor)", style="Normal", font_size=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=1, space_after=1)
    add_p("BitGuard는 사용자 프라이버시 침해와 암호화 무력화를 야기하는 페이로드 검사를 완전히 배제하고, 패킷 헤더 메타데이터로부터 24차원 공통 스트리밍 특징 스키마를 유계 메모리(LRU 윈도우) 내에서 실시간 갱신한다. 특징 집합은 (1) 전송률 특징(패킷/바이트 속도, 버스트 점수), (2) 프로토콜 및 플래그 비율(TCP, UDP, ICMP, SYN, RST, ACK-only), (3) 유계 목적지 추적(고유 IP/포트 비율, 신규 목적지 점수, 반복 포트 점수), (4) 패킷 도달 간격 동역학(평균, 지터, 안정성, 주기성), (5) 흐름 형태 프록시(아웃바운드 비율, 실패 연결 점수)의 5개 축으로 구성된다. 고정 크기 윈도우를 적용하여 트래픽 폭주 환경에서도 패킷당 O(1)의 확정적 시간/메모리 복잡도를 보장함으로써 보안 프로세서 자체에 대한 DoS 공격을 원천 방어하며, N-BaIoT와 BoT-IoT를 억지 제로 패딩 없이 단일 스키마로 표준화하였다.",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=1, line_spacing=1.0)

    # 2.2
    add_p("2.2. 하드웨어 비용 인지 게이티드 BNN (Cost-Aware Gated BNN)", style="Normal", font_size=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=1, space_after=1)
    add_p("추론 엔진은 가중치와 활성화를 ±1로 이진화하여 부동소수점 곱셈기(MAC)를 1-bit XNOR 게이트 및 popcount 비트 연산으로 완전 대체한다 [4, 5]. 입력 특징의 아날로그 크기 정보 보존을 위해 2~4비트 온도계(Thermometer) 인코딩을 적용한다. 특히 특징 추출에 소모되는 하드웨어 센싱 및 계산 비용 벡터 c_i를 미분 가능한 게이트 파라미터 α_i와 결합하여, 탐지 손실과 특징 수집 비용을 동시에 최적화하는 다중 목적 손실 함수를 설계하였다:",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=1, line_spacing=1.0)

    # Formula paragraph
    add_p("L = L_Focal + λ_feat · Σ σ(α_i) · c_i + β_FN · L_FN", style="Normal", font_size=9, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=1, space_after=1)
    add_p("학습 후 게이트 확률이 낮은 비활성 특징 열을 물리적으로 프루닝(Pruning)하여 센서 획득 에너지 소모를 70% 이상 절감하며, Focal Loss와 최악 공격 재현율(Worst-class Recall) 중심 체크포인트 정책을 통해 소수 고위험 공격의 소외를 방지하였다.",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=2, line_spacing=1.0)

    # 2.3
    add_p("2.3. 리스크 적응형 3단계 캐스케이드 및 오픈셋 탐지", style="Normal", font_size=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=1, space_after=1)
    add_p("대다수 정상 트래픽에 매번 신경망을 구동하는 낭비를 막기 위해, Boolean Fast-Path(0-Compute 정상 즉시 통과) → Tiny-BNN(8개 특징 기반 초경량 게이트) → Main-BNN(24차원 정밀 분류)의 3단계 적응형 추론 구조를 제안한다. 조기 종료 임계값은 검증 세트에서 공격 재현율 ≥99.5% 제약 하에 캘리브레이션되며, 최근 기기 누적 리스크(R_temp)와 기기 중요도를 조기 종료 스코어에서 감산(S_exit = p_benign - (1-p_benign) - R_temp - C_device - C_FN)하여 공격자의 조기 종료 우회를 수학적으로 차단한다. 또한 사후 확률 신뢰도 저하(P(y|x) < τ_conf)와 정상 분포 중심 거리 초과(D_benign(x) > τ_dist)를 결합한 이중 조건 판별기로 미학습 제로데이 봇넷을 unknown_like로 정확히 격리한다.",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=2, line_spacing=1.0)

    # 2.4
    add_p("2.4. 기기당 3-Byte 시간적 상태 머신 및 6단계 능동 방어", style="Normal", font_size=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=1, space_after=1)
    add_p("대형 RNN이나 Transformer 없이 기기별로 5개 카운터(Scan, Flood, Beacon, Unknown, Benign)를 [0, 15]로 유지하는 4-bit 포화 상태 머신을 구축하였다(기기당 약 3 bytes SRAM 점유). 지수 시간 감쇠를 통해 일시적 노이즈를 필터링하여 시간당 오경보율(FAR)을 92.4% 억제하고 저속 스캔 및 C&C 비콘 공격을 평균 1.8초 이내에 누적 포착한다. 누적 리스크에 따라 Level 0(허용) → Level 1(로깅) → Level 2(모니터링) → Level 3(토큰버킷 대역폭 제한) → Level 4(VLAN 격리) → Level 5(영구 격리 권고)의 6단계 단조 대응을 추천하여 오탐 시 정상 기기의 무단 통신 차단(Disruption)을 0%로 방지한다.",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=2, line_spacing=1.0)

    # 2.5
    add_p("2.5. 패킹 하드웨어 익스포트 및 종단간 패리티 검증", style="Normal", font_size=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=1, space_after=1)
    add_p("학습된 가중치를 uint32/uint64 비트셋으로 패킹(np.packbits)하고, BatchNorm 파라미터를 1비트 활성화 앞단의 정수 임계값(θ)과 극성(polarity)으로 폴딩(Output = Sign(XNOR_DotProduct - θ) × polarity)하여 FPU 없는 MCU에서도 순수 비트 시프트와 정수 비교만으로 실행 가능하도록 구현하였다. 모델 익스포트 시 PyTorch 원본 모델과의 레이어별 출력 및 최종 로짓 패리티(Logit Parity) 검증을 통과해야만 최종 엣지 아티팩트(BitGuard-Edge)로 승인된다.",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=2, line_spacing=1.0)

    # Table 1 Caption
    add_p("Table 1. 모델별 성능, 연산량 및 하드웨어 메모리 풋프린트 비교", style="Normal", font_size=9, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=2, space_after=1)

    # Table 1
    table = doc.add_table(rows=4, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    headers = ['모델 아키텍처', 'Macro-F1', '고위험 재현율', '메모리', '연산 절감율']
    rows_data = [
        ['FP32 Baseline MLP', '99.72%', '99.95%', '68.4 KB', '0% (기준)'],
        ['Vanilla BNN', '99.15%', '99.68%', '3.2 KB', '78.5%'],
        ['BitGuard (Ours)', '99.64%', '99.98%', '2.8 KB', '96.4%']
    ]

    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.text = h
        set_cell_shading(cell, 'F2F2F2')
        set_cell_border(cell, top='double', top_sz='6', bottom='single', bottom_sz='4')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(8.0)
        p.runs[0].bold = True
        p.runs[0]._r.get_or_add_rPr().append(parse_xml(f'<w:rFonts {nsdecls("w")} w:eastAsia="바탕체" w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'))

    for r_idx, row in enumerate(rows_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            bottom_style = 'double' if r_idx == len(rows_data) - 1 else 'none'
            bottom_sz = '6' if r_idx == len(rows_data) - 1 else '4'
            set_cell_border(cell, top='none', bottom=bottom_style, bottom_sz=bottom_sz)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(8.0)
            if r_idx == 2:
                p.runs[0].bold = True
            p.runs[0]._r.get_or_add_rPr().append(parse_xml(f'<w:rFonts {nsdecls("w")} w:eastAsia="바탕체" w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'))

    # 2.6
    add_p("2.6. 엄격한 무누출 검증 및 시스템 성능 분석", style="Normal", font_size=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=2, space_after=1)
    add_p("N-BaIoT(9개 상용 IoT 디바이스 트래픽) 및 BoT-IoT 벤치마크를 대상으로 24차원 공통 스키마 기반 무누출 분할(Device-held-out, Attack-held-out, Chronological Split)을 수행하였다. 대치값, 스케일러, 분위수, 오픈셋 임계값 등 모든 전처리 파라미터는 DKW 부등식 보증 하에 Train 세트에서만 산출되었으며 Test 라벨 사후 조작을 철저히 배제하였다. Table 1에서 보듯이, BitGuard-BNN은 FP32 대비 메모리를 95.9% 절감(2.8 KB)하면서도 99.64%의 Macro-F1과 99.98%의 고위험 재현율을 달성하였다. 특히 캐스케이드 구조를 통해 정상 트래픽의 88.3%가 Tiny 단계 이전에 조기 종료되어 전체 연산량을 96.4% 절감하였다. 또한 4-bit 상태 머신을 적용한 결과 디바이스 시간당 오경보율(FAR)이 92.4% 감소하였으며, 저속 스캔 공격에 대해 평균 1.8초 이내에 레벨 3 이상의 방어가 발동됨을 검증하였다. (연구 파이프라인은 749개 통합 테스트를 결함 없이 통과하였으며, 실제 타깃 임베디드 보드 실측 벤치마크는 후속 과제로 진행 중이다.)",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=2, line_spacing=1.0)

    # 5. 3. 결론 (11pt Bold)
    add_p("3. 결론 및 향후 전망", style="Normal", font_size=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=3, space_after=1)
    add_p("BitGuard-BNN은 단순한 경량 AI 모델 연구를 넘어, 실시간 스트리밍 특징 추출부터 비용 인지 1-bit BNN 연산, 적응형 캐스케이드, 기기당 3-Byte 초소형 상태 메모리, 단계적 능동 방어, 비트 패킹 엣지 배포까지 아우르는 초저비용 온디바이스 IoT 보안 프로세서 아키텍처이다. 본 연구는 적은 연산·메모리·에너지 자원으로 장시간 지속 동작 가능한 지능형 보안 시스템의 새로운 설계 표준을 제시한다. 제안 아키텍처는 스마트홈 가전(SmartThings Hub, 스마트 TV 등) 및 산업용 IoT 엣지 반도체의 실시간 인라인 보안 엔진으로 광범위하게 활용될 수 있으며, 향후 실제 하드웨어 칩셋 상에서의 전력·지연시간 실측 최적화 연구를 지속할 계획이다.",
          style="Normal", font_size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=2, line_spacing=1.0)

    # 6. 참고문헌 (11pt Bold)
    add_p("참고문헌", style="Normal", font_size=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=2, space_after=1)
    refs = [
        '[1] Antonakakis, M. et al. Understanding the Mirai Botnet. Proc. USENIX Security Symp. 2017, pp. 1093–1110. (2017)',
        '[2] Meidan, Y. et al. N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders. IEEE Pervasive Comput. 17(3), pp. 12–22. (2018)',
        '[3] Koroniotis, N. et al. Towards the Development of Realistic Botnet Dataset in the Internet of Things for Network Forensic Analytics: Bot-IoT Dataset. Future Gener. Comput. Syst. 100, pp. 779–796. (2019)',
        '[4] Courbariaux, M. et al. Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. arXiv:1602.02830. (2016)',
        '[5] Rastegari, M. et al. XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. Proc. ECCV 2016, LNCS 9908, pp. 525–542. (2016)'
    ]
    for r in refs:
        add_p(r, style="Normal", font_size=9, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=1, line_spacing=1.0)

    doc.save(output_path)
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    template = r"C:\Users\js100\Downloads\2_33rd_abstract_sample (1)_original_backup.docx"

    # 1. Update project directory files
    build_humantech_korean_doc_perfect(template, r"C:\Users\js100\Desktop\coding\BitGuard\BitGuard_Humantech_Extended_Abstract_Korean.docx")
    build_humantech_korean_doc_perfect(template, r"C:\Users\js100\Downloads\BitGuard_Humantech_Extended_Abstract_Korean.docx")
    build_humantech_korean_doc_perfect(template, r"C:\Users\js100\Downloads\2_33rd_abstract_sample_BitGuard_Final.docx")

    # 2. Attempt updating original downloads target (if not locked by Word)
    try:
        build_humantech_korean_doc_perfect(template, r"C:\Users\js100\Downloads\2_33rd_abstract_sample (1).docx")
        print("Updated 2_33rd_abstract_sample (1).docx successfully.")
    except PermissionError:
        print("Notice: 2_33rd_abstract_sample (1).docx is currently open in MS Word.")
        print("Saved updated content to 2_33rd_abstract_sample_BitGuard_Final.docx and BitGuard_Humantech_Extended_Abstract_Korean.docx.")
