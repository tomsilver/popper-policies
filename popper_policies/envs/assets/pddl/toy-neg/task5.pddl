(define (problem toy-delivery) (:domain toy-delivery)
  (:objects
    loc0 loc1 loc2 loc3 loc4
  )
  (:init 
	(at-robby loc2)
  (satisfied loc0)
  (satisfied loc1)
  (satisfied loc3)
  )
  (:goal (and (satisfied loc0)  (satisfied loc1)  (satisfied loc3)  (satisfied loc4)))
)
