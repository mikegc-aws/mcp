# Vibe coding tips and tricks

> Note
> This field is evolving quickly, and we will update this guide as new methods and recommendations arise.

## Table of Contents

- [Vibe coding](#vibe-coding)
- [Requirements and Design Guidelines](#requirements-and-design-guidelines)
- [Prompting](#prompting)
- [Testing and Validation](#testing-and-validation)
- [Context](#context)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [Version Control](#version-control)

## Vibe coding

As described [here](https://en.wikipedia.org/wiki/Vibe_coding), vibe coding is a modern approach to software development where users enter prompts in natural language to generate code.

Vibe coding involves several key components working together:

- **Prompt**: The initial instructions and context provided to guide the coding process
- **Client**: The interface through which users interact with the coding system. For instance, [Amazon Q Developer](https://aws.amazon.com/q/developer/) or [Cline](https://cline.bot/)
- **Additional context**: You can enhance the agent's capabilities by providing additional context, such as AWS MCP servers

> Warning
> Never blindly trust code generated by AI assistants. Always:
>
> - Thoroughly review and understand the generated code
> - Verify all dependencies
> - Perform necessary security checks
> - Test the code in a controlled environment

## Requirements and Design Guidelines

Before starting any coding task, follow these essential steps:

- Clearly define project requirements and scope
- Establish comprehensive design guidelines and coding standards
- Document all constraints and limitations
- Create and maintain markdown files with gathered information for client access
- Begin the coding process only after completing the above steps

## Prompting

Effective prompting is crucial for successful AI-assisted development:

- Provide detailed specifications for the work to be done
- Include relevant context and files when necessary
- Apply prompting strategically to specific tasks for easier review and testing
- Break down large tasks into smaller, focused subtasks for better results

## Testing and Validation

Ensure code quality and reliability through:

- Incremental testing of each change
- Implementation of automated testing where possible
- Validation against original requirements
- Maintenance of a comprehensive test suite (CI/CD)
- Regular automated security and quality scans

## Documentation

Maintain high-quality documentation by:

- Documenting every change made
- Ensuring the client generates appropriate code documentation (e.g., Python docstrings in code and written documentation in a README)
- Keeping documentation up-to-date with code changes

## Limitations

### Number of MCP servers and tools

- Excessive number of MCP servers/tools can negatively impact client performance
- Refer to your client documentation for best practices and limits

### Conversation Management

- Long conversations can degrade client performance due to growing context size
- Maintain separate conversations for different features
- Regularly review and clean up conversation history

## Context

### Rules and Configuration

For optimal results:

- Define clear rules for code generation and modification
- Maintain consistent configuration across environments
- Document special rules and exceptions
- Regularly review and update configuration settings
- Implement modular design principles

## Version Control

Follow these version control best practices:

- Commit changes frequently with meaningful messages
- Use feature branches for new development
- Maintain a clean and organized repository structure
