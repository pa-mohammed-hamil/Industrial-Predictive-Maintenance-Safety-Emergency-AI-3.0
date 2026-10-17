# app/integrations/notifications/email.py

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmailConfig:
    host: str = "localhost"
    port: int = 587

    username: str | None = None
    password: str | None = None

    sender_email: str = "alerts@industrial-ai.local"
    sender_name: str = "Industrial AI"

    use_tls: bool = True
    use_ssl: bool = False

    timeout: float = 15.0

    emergency_recipients: list[str] = field(default_factory=list)
    safety_recipients: list[str] = field(default_factory=list)
    maintenance_recipients: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EmailAttachment:
    filename: str
    content: bytes
    content_type: str = "application/octet-stream"


class EmailClient:
    """SMTP notification client for the Industrial AI platform."""

    def __init__(self, config: EmailConfig) -> None:
        self.config = config

        if config.use_ssl and config.use_tls:
            raise ValueError(
                "use_ssl and use_tls cannot both be enabled"
            )

    async def send(
        self,
        *,
        recipients: str | Sequence[str],
        subject: str,
        text: str,
        html: str | None = None,
        cc: Sequence[str] | None = None,
        bcc: Sequence[str] | None = None,
        attachments: Sequence[EmailAttachment] | None = None,
    ) -> None:
        to_addresses = self._normalize_recipients(recipients)
        cc_addresses = self._normalize_recipients(cc or [])
        bcc_addresses = self._normalize_recipients(bcc or [])

        if not to_addresses:
            raise ValueError("At least one email recipient is required")

        message = EmailMessage()

        message["From"] = formataddr(
            (
                self.config.sender_name,
                self.config.sender_email,
            )
        )
        message["To"] = ", ".join(to_addresses)
        message["Subject"] = subject

        if cc_addresses:
            message["Cc"] = ", ".join(cc_addresses)

        message.set_content(text)

        if html:
            message.add_alternative(
                html,
                subtype="html",
            )

        for attachment in attachments or []:
            self._add_attachment(message, attachment)

        all_recipients = list(
            dict.fromkeys(
                to_addresses
                + cc_addresses
                + bcc_addresses
            )
        )

        await asyncio.to_thread(
            self._send_sync,
            message,
            all_recipients,
        )

    def _send_sync(
        self,
        message: EmailMessage,
        recipients: list[str],
    ) -> None:
        try:
            if self.config.use_ssl:
                context = ssl.create_default_context()

                with smtplib.SMTP_SSL(
                    self.config.host,
                    self.config.port,
                    timeout=self.config.timeout,
                    context=context,
                ) as server:
                    self._authenticate(server)
                    server.send_message(
                        message,
                        to_addrs=recipients,
                    )

            else:
                with smtplib.SMTP(
                    self.config.host,
                    self.config.port,
                    timeout=self.config.timeout,
                ) as server:

                    server.ehlo()

                    if self.config.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                        server.ehlo()

                    self._authenticate(server)

                    server.send_message(
                        message,
                        to_addrs=recipients,
                    )

            logger.info(
                "Email sent successfully recipients=%s subject=%s",
                recipients,
                message["Subject"],
            )

        except (
            smtplib.SMTPException,
            OSError,
        ):
            logger.exception(
                "Failed sending email subject=%s",
                message["Subject"],
            )
            raise

    def _authenticate(
        self,
        server: smtplib.SMTP,
    ) -> None:
        if not self.config.username:
            return

        if not self.config.password:
            raise ValueError(
                "SMTP password required when username is configured"
            )

        server.login(
            self.config.username,
            self.config.password,
        )

    @staticmethod
    def _normalize_recipients(
        recipients: str | Iterable[str],
    ) -> list[str]:

        if isinstance(recipients, str):
            recipients = [recipients]

        result: list[str] = []

        for recipient in recipients:
            value = recipient.strip()

            if value:
                result.append(value)

        return result

    @staticmethod
    def _add_attachment(
        message: EmailMessage,
        attachment: EmailAttachment,
    ) -> None:

        try:
            maintype, subtype = attachment.content_type.split(
                "/",
                maxsplit=1,
            )
        except ValueError:
            maintype = "application"
            subtype = "octet-stream"

        message.add_attachment(
            attachment.content,
            maintype=maintype,
            subtype=subtype,
            filename=attachment.filename,
        )

    # --------------------------------------------------
    # Industrial notification helpers
    # --------------------------------------------------

    async def send_safety_alert(
        self,
        *,
        site_id: str,
        asset_id: str,
        event_type: str,
        severity: str,
        description: str,
        recipients: Sequence[str] | None = None,
    ) -> None:

        targets = list(
            recipients or self.config.safety_recipients
        )

        if not targets:
            logger.warning(
                "Safety email skipped: no recipients configured"
            )
            return

        subject = (
            f"[{severity.upper()}] Safety Alert - "
            f"{asset_id}: {event_type}"
        )

        text = (
            "Industrial Safety Alert\n\n"
            f"Site: {site_id}\n"
            f"Asset: {asset_id}\n"
            f"Event: {event_type}\n"
            f"Severity: {severity.upper()}\n\n"
            f"Description:\n{description}\n"
        )

        html = f"""
        <h2>Industrial Safety Alert</h2>

        <table>
            <tr>
                <td><strong>Site</strong></td>
                <td>{site_id}</td>
            </tr>
            <tr>
                <td><strong>Asset</strong></td>
                <td>{asset_id}</td>
            </tr>
            <tr>
                <td><strong>Event</strong></td>
                <td>{event_type}</td>
            </tr>
            <tr>
                <td><strong>Severity</strong></td>
                <td>{severity.upper()}</td>
            </tr>
        </table>

        <h3>Description</h3>

        <p>{description}</p>
        """

        await self.send(
            recipients=targets,
            subject=subject,
            text=text,
            html=html,
        )

    async def send_emergency_alert(
        self,
        *,
        site_id: str,
        asset_id: str,
        incident_id: str,
        emergency_type: str,
        description: str,
        recipients: Sequence[str] | None = None,
    ) -> None:

        targets = list(
            recipients or self.config.emergency_recipients
        )

        if not targets:
            logger.warning(
                "Emergency email skipped: no recipients configured"
            )
            return

        subject = (
            f"[EMERGENCY] {emergency_type} - "
            f"{site_id}/{asset_id}"
        )

        text = (
            "INDUSTRIAL EMERGENCY ALERT\n\n"
            f"Incident ID: {incident_id}\n"
            f"Site: {site_id}\n"
            f"Asset: {asset_id}\n"
            f"Emergency: {emergency_type}\n\n"
            f"{description}\n\n"
            "Immediate investigation may be required."
        )

        await self.send(
            recipients=targets,
            subject=subject,
            text=text,
        )

    async def send_maintenance_alert(
        self,
        *,
        site_id: str,
        asset_id: str,
        failure_probability: float,
        rul_hours: float | None,
        recommendation: str,
        recipients: Sequence[str] | None = None,
    ) -> None:

        targets = list(
            recipients or self.config.maintenance_recipients
        )

        if not targets:
            logger.warning(
                "Maintenance email skipped: no recipients configured"
            )
            return

        probability = max(
            0.0,
            min(float(failure_probability), 1.0),
        )

        subject = (
            f"Predictive Maintenance Alert - {asset_id}"
        )

        rul_text = (
            f"{rul_hours:.1f} hours"
            if rul_hours is not None
            else "Unavailable"
        )

        text = (
            "Predictive Maintenance Alert\n\n"
            f"Site: {site_id}\n"
            f"Asset: {asset_id}\n"
            f"Failure probability: {probability:.1%}\n"
            f"Estimated RUL: {rul_text}\n\n"
            f"Recommendation:\n{recommendation}"
        )

        await self.send(
            recipients=targets,
            subject=subject,
            text=text,
        )

    async def send_report(
        self,
        *,
        recipients: Sequence[str],
        subject: str,
        report_path: str | Path,
        message: str = "Please find the attached report.",
    ) -> None:

        path = Path(report_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Report not found: {path}"
            )

        attachment = EmailAttachment(
            filename=path.name,
            content=await asyncio.to_thread(
                path.read_bytes
            ),
            content_type=self._guess_content_type(path),
        )

        await self.send(
            recipients=recipients,
            subject=subject,
            text=message,
            attachments=[attachment],
        )

    @staticmethod
    def _guess_content_type(path: Path) -> str:
        types = {
            ".pdf": "application/pdf",
            ".csv": "text/csv",
            ".json": "application/json",
            ".txt": "text/plain",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }

        return types.get(
            path.suffix.lower(),
            "application/octet-stream",
        )


def create_email_client(
    *,
    host: str,
    port: int,
    sender_email: str,
    sender_name: str = "Industrial AI",
    username: str | None = None,
    password: str | None = None,
    use_tls: bool = True,
    use_ssl: bool = False,
) -> EmailClient:
    """Factory for application startup/dependency injection."""

    config = EmailConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        sender_email=sender_email,
        sender_name=sender_name,
        use_tls=use_tls,
        use_ssl=use_ssl,
    )

    return EmailClient(config)