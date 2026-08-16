#!/bin/bash
# =============================================================================
# 3-SEED × 4-CONFIG × 6-FOLD SWEEP CHO PAPER
# =============================================================================
#
# Chạy 3 seed độc lập (42, 43, 44), mỗi seed có OUT_DIR riêng để checkpoint
# không lẫn nhau. Mỗi seed sẽ tạo 24 checkpoint (4 config × 6 fold) trên disk.
# Tổng: 72 checkpoint = 3 × 24.
#
# Ước tính thời gian trên Colab Pro T4:
#   ~400s/fold × 4 config × 6 fold × 3 seed = ~8 giờ
#   Kèm probe LR + MLP + permutation × 24 fold ≈ thêm 30-60 phút
#   → total ~9 giờ
#
# CHIA THÀNH 3 SESSION Colab (mỗi session ~3h):
#   Session 1: seed 42 (đã có? nếu có --force để chạy lại với code mới đã fix)
#   Session 2: seed 43
#   Session 3: seed 44
#
# RESUME: nếu Colab disconnect giữa chừng, chạy lại cùng lệnh — checkpoint đã có
#   sẽ được skip, chỉ chạy tiếp fold/config còn thiếu.
#
# CÁCH DÙNG (chạy trên Colab hoặc terminal):
#   bash run_seed_sweep.sh 42     # chạy chỉ seed 42
#   bash run_seed_sweep.sh 43     # chạy seed 43
#   bash run_seed_sweep.sh all    # chạy tuần tự cả 3 seed (dùng khi có thời gian dài)
#
# TRÊN COLAB:
#   !bash run_seed_sweep.sh 42
# =============================================================================

set -e

# ── Thư mục làm việc: có thể override qua env WORK ───────────────────────────
: "${WORK:=/content/drive/MyDrive/ravdess_experiment}"
BASE_OUT_DIR="$WORK/outputs_speaker_adversarial"

run_one_seed() {
    local SEED=$1
    local OUT_DIR="${BASE_OUT_DIR}/seed_${SEED}"

    echo ""
    echo "############################################################"
    echo "  SEED $SEED  →  $OUT_DIR"
    echo "############################################################"
    mkdir -p "$OUT_DIR"

    # Chạy tuần tự 4 config. Nhờ cơ chế checkpoint bền vững, có thể chạy
    # gộp 1 lệnh (--configs base aug grl aug+grl) hoặc tách 4 lệnh — kết
    # quả như nhau. Tách ra để in tổng hợp giữa các bước, dễ theo dõi.
    for CFG in base aug grl aug+grl; do
        echo ""
        echo ">>> SEED=$SEED  CONFIG=$CFG"
        SEED="$SEED" OUT_DIR="$OUT_DIR" python speaker_adversarial.py --configs "$CFG"
    done

    echo ""
    echo "✓ SEED $SEED HOÀN THÀNH — checkpoint tại $OUT_DIR"
}

case "${1:-help}" in
    42|43|44)
        run_one_seed "$1"
        ;;
    all)
        for S in 42 43 44; do
            run_one_seed "$S"
        done
        echo ""
        echo "############################################################"
        echo "  ALL 3 SEEDS DONE — gộp bằng merge_seeds.py để có stats"
        echo "############################################################"
        ;;
    *)
        echo "Cách dùng: bash run_seed_sweep.sh [42|43|44|all]"
        echo "  42/43/44: chạy 1 seed cụ thể (khuyên dùng khi lo Colab disconnect)"
        echo "  all: chạy tuần tự cả 3 seed (~9 giờ liên tục)"
        exit 1
        ;;
esac
