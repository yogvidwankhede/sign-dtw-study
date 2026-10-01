# Ethics, scope and community feedback

## Scope statement (used in the paper)

This work evaluates isolated-sign recognition from hand and body landmarks on one dataset of American Sign Language
recorded on smartphones. It is not sign language translation or interpretation, it does not model facial or other
non-manual grammar, and it makes no claim about usefulness to Deaf people. Results on PopSign (47 signers, 250 signs,
one-handed focus, selfie camera) do not transfer automatically to other signers, signs, sign languages or camera setups.

## Who did the work

The study was designed and run by hearing researchers without a Deaf co-author. Prior Deaf-led work argues that this
pattern, and a framing around "communication barriers", shape research agendas in ways that do not serve Deaf communities
(Desai et al. 2024). The paper should say plainly that this study is subject to that critique, and why it was done anyway:
it is a methods comparison on an existing consented dataset, not an application.

## Data

PopSign participants consented to inclusion in a public database; the dataset card lists identification and appearance-based
uses as out of scope. We keep face landmarks local and unreleased, release only hand and pose landmarks, and never try to
identify signers. ASL Citizen's licence forbids redistribution; any use of it stays private and is deleted afterwards.

## Risks and how they are handled

- Overclaiming: no "interpreter", "translator" or "understands" wording; the abstract and limitations state the scope.
- Misuse of a cheap recogniser as a substitute for interpreters or for language access: the discussion says recognition
  accuracy on isolated signs is not communication access, and cites the community-perspective literature.
- Surveillance or biometric reuse of landmarks: face landmarks are not released; the released files contain no video.

## Getting feedback from Deaf community members before submission

The author must do this; it cannot be done by the code. Suggested route:
1. Share the two-page scope statement and the abstract with Deaf researchers or a local Deaf community organisation, and
   ask for comments on framing, the "who benefits" language, and anything that reads as overclaiming.
2. Offer paid time for reviewers; do not ask for unpaid labour.
3. Ask specific questions: Is the scope statement clear? Does anything in the discussion imply benefit we cannot show?
   Is there prior work or a community position we missed? Would you want this released in this form?
4. Record who was asked, what changed because of their comments, and what was not changed and why (acknowledge by name only
   with consent).
5. If no feedback can be obtained before the deadline, say so in the paper instead of implying consultation.

Draft outreach message (edit before sending):

> Hello, I am a student researcher at Washington University in St. Louis. I am finishing a small methods paper that compares
> template-matching and neural models for recognising isolated ASL signs from hand landmarks on the PopSign dataset. It is
> not a translation system and I do not claim it helps Deaf users. Because I am not Deaf and the work was not Deaf-led, I would
> value your critique of how I describe its scope and limits before I submit it. I can send a two-page summary, and I would
> like to compensate you for your time. Would you be willing to read it?
