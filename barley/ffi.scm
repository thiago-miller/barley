(define-module (barley ffi)
  #:use-module (system foreign)
  #:use-module (system foreign-library)
  #:use-module (rnrs bytevectors)
  #:export (%bcl-env-alloc %bcl-env-free %bcl-error
            %bcl-tensor-alloc %bcl-tensor-free
            call-with-bcl-error
            BCL_DEVICE_TYPE_ANY BCL_DEVICE_TYPE_CPU BCL_DEVICE_TYPE_GPU))

(define BCL_DEVICE_TYPE_ANY 0)
(define BCL_DEVICE_TYPE_CPU 1)
(define BCL_DEVICE_TYPE_GPU 2)

(define libbcl (load-foreign-library "libbcl"))

(define-syntax-rule (define-bcl name c-name ret (arg ...))
  (define name
    (foreign-library-function libbcl c-name
                              #:return-type ret
                              #:arg-types (list arg ...))))

(define-bcl %bcl-strerror     "bcl_strerror"     '*   (int))
(define-bcl %bcl-env-alloc    "bcl_env_alloc"    '*   (int '*))
(define-bcl %bcl-env-free     "bcl_env_free"     void ('*))
(define-bcl %bcl-tensor-alloc "bcl_tensor_alloc" '*   ('* size_t '* '* int '*))
(define-bcl %bcl-tensor-free  "bcl_tensor_free"  void ('*))

(define (%bcl-error err)
  (pointer->string (%bcl-strerror err)))

;; PROC receives a pointer to an int to use as `BclError *out_err`.
;; Throws 'bcl-error if the C side reported an error or returned NULL.
(define (call-with-bcl-error who proc)
  (let* ((bv (make-bytevector 4 0))
         (result (proc (bytevector->pointer bv)))
         (err (bytevector-s32-native-ref bv 0)))
    (when (or (not (zero? err))
              (and (pointer? result) (null-pointer? result)))
      (throw 'bcl-error who err (%bcl-error err)))
    result))
