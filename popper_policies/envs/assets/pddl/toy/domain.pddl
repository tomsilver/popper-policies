(define (domain toy-delivery)
    (:requirements :strips)
    (:predicates 
        (at-robby ?loc)
        (satisfied ?loc)
    )
    
    (:action move
        :parameters (?from ?to)
        :precondition (and
            (at-robby ?from) 
        )
        :effect (and
            (not (at-robby ?from))
            (at-robby ?to)
        )
    )
    
    (:action deliver
        :parameters (?loc)
        :precondition (and
            (at-robby ?loc)
        )
        :effect (and
            (satisfied ?loc)
        )
    )
    
)