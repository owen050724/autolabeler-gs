# sample_images

이 폴더에 자동 라벨링을 시험할 이미지(JPG / PNG 등)를 넣어주세요.

## 데모 이미지 생성

저작권 걱정 없이 빠르게 시험해보려면 합성 이미지를 만들 수 있습니다.

```bash
python scripts/make_demo_assets.py
```

위 스크립트는 `sample_images/` 폴더에 단순 도형으로 구성된 `demo_*.png` 4장을
생성합니다. 그런 다음 다음과 같이 실행해보세요.

```bash
python -m autolabeler.cli \
  --images sample_images \
  --classes "rectangle, circle" \
  --out runs/demo \
  --mock
```

## 권장 이미지

- 해상도 640x480 ~ 1920x1080 정도가 무난합니다.
- 클래스 프롬프트는 영어가 검출 성능이 더 좋습니다.
- 사진 사용 시 라이선스에 유의해주세요.
