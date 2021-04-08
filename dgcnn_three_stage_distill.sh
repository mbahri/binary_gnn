# Three-stage distillation template by Mehdi Bahri
# m.bahri@imperial.ac.uk

MODEL="$1"
STARTING_POINT="$2"
KNN_OP="$3"
LSP_BIN_METRIC="$4"
LSP_LAM="$5"
KD_T="$6"
KD_alpha="$7"
PREFIX="$8"
BS="$9"
TBS="${10}"
ADDITIONAL_SUFFIX="${11}"
ADDITIONAL_ARGS="${12}"

CKPT_PATH="checkpoints"

# Generate identifiable suffix
SUFFIX=_lsp_${LSP_LAM}_knn_${KNN_OP}_lsp_${LSP_BIN_METRIC}${ADDITIONAL_SUFFIX}

DISTILL_COMMON_ARGS="--LSP_lambda=${LSP_LAM} --KD_T=${KD_T} --KD_alpha=${KD_alpha}"
HYPER_COMMON_ARGS="--bin_knn_op=${KNN_OP}"
MISC_ARGS="--batch_size ${BS} --test_batch_size ${TBS} --dropout 0.5 --bin_bn_momentum 0.999"
COMMON_ARGS="${MISC_ARGS} ${DISTILL_COMMON_ARGS} ${HYPER_COMMON_ARGS} ${ADDITIONAL_ARGS} --suffix $SUFFIX --additional_suffix $ADDITIONAL_SUFFIX"

# Stage one: distill base dgcnn to binary-like model with fp32 weights/activations
STAGE_ONE_NAME="${MODEL}_stage1_distill${SUFFIX}"
STAGE_ONE_DISTILL="--student_model ${MODEL} --teacher_model dgcnn  --teacher_knn_op=l2 --LSP_kernels=l2_l2 --teacher_quantize_weights=False --teacher_quantize_inputs=False --teacher_pseudo_quantize=False"
STAGE_ONE_QUANTIZE="--bin_quantize_weights=False --bin_quantize_inputs=False --bin_pseudo_quantize=True"
STAGE_ONE="--lr 1e-3 --wd 1e-5 ${STAGE_ONE_QUANTIZE} --exp_name ${STAGE_ONE_NAME} --teacher_path ${CKPT_PATH}/starting_points/${STARTING_POINT}/models/model.t7 ${STAGE_ONE_DISTILL} --epochs 250 --stage 1"

# Stage two: distill the net with pseudo-quantized inputs (tanh) and real weights into the net with real weights but binary activations
STAGE_TWO_NAME="${MODEL}_stage2_distill${SUFFIX}"
STAGE_TWO_DISTILL="--init_student_with_teacher_weights=True --student_model ${MODEL} --teacher_model ${MODEL} --teacher_knn_op=l2 --LSP_kernels=${LSP_BIN_METRIC}_l2 --teacher_quantize_weights=False --teacher_quantize_inputs=False --teacher_pseudo_quantize=True"
STAGE_TWO_QUANTIZE="--bin_quantize_weights=False --bin_quantize_inputs=True --bin_pseudo_quantize=False"
STAGE_TWO="--lr 2.5e-4 --wd 1e-5 ${STAGE_TWO_QUANTIZE} --exp_name ${STAGE_TWO_NAME} --teacher_path ${CKPT_PATH}/${STAGE_ONE_NAME}/models/model.t7 ${STAGE_TWO_DISTILL} --epochs 350 --stage 2"

# Stage three: distill the net with real weights and binary activations into the net with binary weights and activations
STAGE_THREE_NAME="${MODEL}_stage3_distill${SUFFIX}"
STAGE_THREE_DISTILL="--init_student_with_teacher_weights=True --student_model ${MODEL} --teacher_model ${MODEL} --teacher_knn_op=${KNN_OP} --LSP_kernels=${LSP_BIN_METRIC}_${LSP_BIN_METRIC} --teacher_quantize_weights=False --teacher_quantize_inputs=True --teacher_pseudo_quantize=False"
STAGE_THREE_QUANTIZE="--bin_quantize_weights=True --bin_quantize_inputs=True --bin_pseudo_quantize=False"
STAGE_THREE="--lr 1e-3 --wd 0 ${STAGE_THREE_QUANTIZE} --exp_name ${STAGE_THREE_NAME} --teacher_path ${CKPT_PATH}/${STAGE_TWO_NAME}/models/model.t7 ${STAGE_THREE_DISTILL} --epochs 350 --stage 3 --scheduler=step"

# Start the stuff
export $PREFIX

python dgcnn_distill.py ${COMMON_ARGS} ${STAGE_ONE}
python dgcnn_distill.py ${COMMON_ARGS} ${STAGE_TWO}
python dgcnn_distill.py ${COMMON_ARGS} ${STAGE_THREE}
