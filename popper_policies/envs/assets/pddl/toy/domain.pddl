(define (domain toy-delivery)
    (:requirements :strips)
    (:predicates 
        (at ?loc)
        (satisfied ?loc)
    )
    
    (:action move
        :parameters (?from ?to)
        :precondition (and
            (at ?from) 
        )
        :effect (and
            (not (at ?from))
            (at ?to)
        )
    )
    
    (:action deliver
        :parameters (?loc)
        :precondition (and
            (at ?loc)
        )
        :effect (and
            (satisfied ?loc)
        )
    )
    
)