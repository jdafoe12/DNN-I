#!/usr/env/bin guile
!#

(use-modules (ice-9 binary-ports))
(use-modules (ice-9 textual-ports))
(use-modules (ice-9 pretty-print))
(use-modules (scheme base))
(load "utils.scm")

(load-mnist training-phase)
(newline) (display "MNIST training set loaded. The training set contains 60,000 labeled handwritten digits, in the form of 28x28 grayscale images.") (newline)

; TRAINING FORWARD PASS.
(define (training-layer-forward input layer activation-function)
(let* ((weights (car layer))
       (biases (cadr layer))
       (activations (cadr (last input)))
       (zl (map + (matrix-*-vector weights activations) biases))
       (new-activations (map activation-function zl)))
  (append input (list (list zl new-activations)))))

(define (training-forward-pass layers input activation-function)
  (if (null? layers)
	(append (remove-last input) (append (list (list (car (car (reverse input))) (softmax (car (car (reverse input))))))))
	(training-forward-pass (cdr layers) 
						   (training-layer-forward input (car layers) activation-function)
						   activation-function)))
; END TRAINING FORWARD-PASS.

;; COMPUTE GRADIENTS.
(define (compute-gradients layers input expected-output activation-function)
  (let* ((model-output (training-forward-pass layers input activation-function))
         (output-layer-output (car (reverse model-output)))
		 (output-error (compute-output-error output-layer-output expected-output activation-function))
		 (output-layer-gradients (compute-gradients-for-layer (cadr (last (remove-last model-output))) output-error))
		 (hidden-layer-gradients (backpropagate-layers (reverse layers) (reverse (remove-last model-output)) output-error activation-function)))
	(reverse (append (list output-layer-gradients) hidden-layer-gradients))))

(define (compute-output-error layer-output expected-output activation-function)
  (let* ((zL (car layer-output))
        (aL (cadr layer-output))
        (jacobian (softmax-derivative zL)) 
        (error (map - aL expected-output)))
        (matrix-*-vector (transpose jacobian) error)))

(define (ReLU-derivative z)
  (if (> z 0) 1 0))

(define (softmax-derivative z)
  (let* ((aL (softmax z))
         (n (length z)))
    (define (derivative i k)
      (if (= i k)
          (* (list-ref aL i) (- 1 (list-ref aL i)))
          (* (- (list-ref aL i)) (list-ref aL k))))
    (define (generate-jacobian i)
      (map (lambda (k) (derivative i k)) (iota n)))
    (map generate-jacobian (iota n))))

(define (compute-gradients-for-layer ai delta)
  (let ((weight-gradients (outer-product delta ai))
        (bias-gradients delta))
       (list weight-gradients bias-gradients)))

(define (backpropagate-layers layers model-output output-error activation-function)
  (define (backpropagate layers model-output err)
    (if (null? (cdr model-output))
        '()
        (let* ((next-layer-output (car model-output))
               (prev-layer-output (if (null? (cdr model-output)) '() (cadr model-output)))
               (next-layer (car layers))
               (weights (car next-layer))
               (biases (cadr next-layer))
               (zi (car next-layer-output))
               (ai (cadr prev-layer-output))
               (delta (compute-delta weights err zi activation-function))
               (gradients (compute-gradients-for-layer ai delta)))
          (cons gradients (backpropagate (cdr layers) (cdr model-output) delta)))))
  (backpropagate layers model-output output-error))

(define (compute-delta weights err zi activation-function)
  (let ((weighted-error (matrix-*-vector (transpose weights) err)))
    (map (lambda (e z)
           (* e (ReLU-derivative z)))
         weighted-error zi)))
;; DONE COMPUTE GRADIENTS.

;; GENERAL TRAINING.
(define (initialize-layer input-size output-size)
  (list
   (map (lambda (_) 
          (map (lambda (_) 
                 (+ 0 (* (sqrt (/ 2 input-size)) (random:normal)))) ; He Initialization
               (iota input-size)))
        (iota output-size))   
   (map (lambda (_)
          (+ 0 (* (sqrt (/ 2 input-size)) (random:normal))))
		(iota output-size))))

(define (train-mnist layers epochs num-steps learning-rate)
  (define (train-loop epoch curr-layers)
    (if (= epoch epochs)
        curr-layers 
        (begin
          (display "Training epoch: ") (display epoch) (newline)
          (let ((new-layers (train-epoch curr-layers num-steps learning-rate)))
            (train-loop (+ epoch 1) new-layers)))))
  (train-loop 0 layers))

(define (train-epoch layers num-steps learning-rate)
  (define (train-step-iter curr-layers remaining-steps)
    (if (= remaining-steps 0)
        curr-layers
		(let* ((input (list (list 0 (load-next-image)))) 
               (expected-output (load-next-label))
			   (new-weights (train-step curr-layers input expected-output learning-rate)))
		  (train-step-iter new-weights (- remaining-steps 1)))))   
  (train-step-iter layers num-steps))

(define (train-step layers input expected-output learning-rate)
  (update-weights layers (compute-gradients layers input expected-output ReLU) learning-rate))

(define (update-weights layers gradients learning-rate)
  (map (lambda (layer gradient) 
		 (list (matrix-minus-matrix (car layer) (matrix-*-scalar (car gradient) learning-rate))
			   (matrix-minus-matrix (cadr layer) (matrix-*-scalar (cadr gradient) learning-rate)))) 
	   layers gradients))
;; DONE GENERAL TRAINING.

;; SPECIFIC TRAINING.
(define layers (list (initialize-layer 784 32)
					 (initialize-layer 32 32)
					 (initialize-layer 32 32)
					 (initialize-layer 32 10)))

(save-model (train-mnist layers 6 10000 0.01) "trained-model")
;; DONE SPECIFIC TRAINING.
