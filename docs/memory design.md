Memory service design
================================================================================

Housekeeping
------------

[ ] - Separate async task.
    [ ] - Has its own queue (with a maxlength to prevent backing up)
    [ ] - Runs on a timer
    [ ] - Only runs when the user has been inactive for a specific period of time
    [ ] - Processes one queued task at a time, then checks again (whether user was recently active)
          This allows it to pause if the user starts interacting again.