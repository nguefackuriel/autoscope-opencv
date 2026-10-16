# Draft: request to AWS for frontier model access on Bedrock

Not sent. The account is in Alex Bruno's name, so the request goes from him,
through the form the Bedrock error message points to:
https://aws.amazon.com/contact-us/sales-support/
(topic: Amazon Bedrock, model access). Fill in the name, company and account
fields yourself; nothing below is private.

Use it only if the runtime is still refusing after the agreements have been
AVAILABLE for an hour. If a model answers before then, this is not needed.

---

**Subject:** Bedrock model access for an OpenCV AI Competition 2026 entry,
account 617847240757

Hello,

We are team AutoScope in the OpenCV AI Competition 2026, powered by AWS. Our
entry is a malaria microscopy pipeline on Graviton with an agent on Amazon
Bedrock that decides which microscope field to read next and when to stop. The
submission deadline is 26 October 2026.

We compare the agent across models. Ten Bedrock models are measured so far,
including Claude Sonnet 4.6, Claude Haiku 4.5, Amazon Nova Pro, GPT-OSS 120B,
DeepSeek v3.2, Mistral Devstral 2 and Nvidia Nemotron. We would like to add
three more and cannot:

- anthropic.claude-opus-5
- anthropic.claude-sonnet-5
- openai.gpt-6-sol

For each, `aws bedrock get-foundation-model-availability` reports
agreementAvailability AVAILABLE, authorizationStatus AUTHORIZED,
entitlementAvailability AVAILABLE and regionAvailability AVAILABLE in us-east-1.
We accepted the usage-based offer for each through
`create-foundation-model-agreement` on 16 October. The Anthropic use case form
is submitted. Yet every Converse call returns:

    AccessDeniedException: anthropic.claude-opus-5 is not available for this
    account. You can explore other available models on Amazon Bedrock. For
    additional access options, contact AWS Sales.

The same message comes back for the global and us inference profiles and for
the bare model id. Claude Sonnet 4.6 and Claude Haiku 4.5 work on the same
account with the same configuration.

The use is research and evaluation only, on public de-identified microscopy
data. Each model arm is about 900 calls and 1.6 million input tokens, under
15 dollars per model at list price. We are not asking for credits, only for
the three models to be callable on this account before 26 October.

Could you tell us what is blocking these three models on this account, and
whether it can be lifted in time?

Thank you,

Alex Bruno Kenfack Anafack, for team AutoScope
AWS account 617847240757
Live demo: https://d1k2o3z37c1maf.cloudfront.net
